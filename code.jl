#' Activate the project and instantiate using the following code
#+ term=false
using Pkg; Pkg.activate(".")
# uncomment the following to install all the required packages
# Pkg.instantiate()

#' load required packages
#+ term=false
using Dates, DataFrames, CSV, JDF, Arrow, GLM, StatsBase, HypothesisTests
using PythonCall, MLJ, XGBoost, FLoops, Geocoder, CairoMakie # GLMakie
import StatsPlots as sp
import Plots as plt

#' + Data cleaning.
#' - For ease of comparison across languages, make the column names consistent
#'   in style with lowercase using underscore to separate words within a name.

#' load data and rename columns
@time dat = CSV.read("nyc311_011523-012123_by022023.csv", DataFrame,
               dateformat="m/d/y H:M:S p")

@time rename!(x -> lowercase(replace(x, " " => "_")), dat)

#' Save and load the data with other formats 

#+ eval=false
# @time Arrow.write("nyc311.arrow", dat)
@time dat = Arrow.Table("nyc311.arrow") |> DataFrame
@time dat = copy(dat)

# @time JDF.save("nyc311.jdf", dat)
@time dat = JDF.load("nyc311.jdf") |> DataFrame

#' - Check for obvious errors or inefficiencies. For example, are there records
#'   whose Closed Date is earlier than or exactly the same as the Created
#'   Date? Are their invalid values for any columns? Are any columns redundant?

first(dat, 5)
last(dat, 5)
n, p = size(dat)
ds = describe(dat)
ds[!, [:variable, :nmissing]] # ds[!, [1, 6]]
ds.mis_rate .= ds.nmissing ./ n
mis_pattern = ds[!, [:variable, :mis_rate]]

#' agency v.s. agency_name, location is redundant with latitude and longitude ...

dat1 = dat[!, [1:4; 6:10; 25:26; 39:40]]
ds1 = describe(dat1)

dat2 = dat1[isless.(dat1.created_date, dat1.closed_date), :]
# dat2 = filter(r -> isless(r.created_date, r.closed_date) ,dat1)

#' - Fill in missing values if possible. For example, if incident zip code is
#'   missing but the location is not, the zip code could be recovered by
#'   geocoding.

idx = findall(ismissing.(dat2.latitude) .& .!ismissing.(dat2.incident_address))

#' Geocoder.jl can be used as a geocoding client for Google Maps API and Open Street Maps API. It requires an APT key to use, so the missing coordiantes are stored in the jdf format.
include("get_coordinates.jl")
coor = geocode("28-07 JACKSON AVENUE, New York", osm_key, "osm")

if abspath(PROGRAM_FILE) == @__FILE__ # only run as the main script
    coordinates = DataFrame([c => Float64[] for c in ["lat", "lng"]])
    ads = dat2.incident_address[idx]
    for a in ads
        g = geocode(a * " New York", osm_key, "osm")
        push!(coordinates, g)
    end
    # CSV.write("coordinates.csv", coordinates)
    # Arrow.write("coordinates.arrow", coordinates)
    JDF.save("coordinates.jdf", coordinates)
else
    coordinates = JDF.load("coordinates.jdf") |> DataFrame
end

rename!(coordinates, [:latitude, :longitude])
dat2[idx, [:latitude, :longitude]] .= coordinates

#' - Summarize your suggestions to the data curator in several bullet points.

#' use a variable dictionary instead of additional columns

#' + Data manipulation. Focus only on requests made to NYPD. 
#' - Create a a new variable `duration`, which represents the time period from
#'   the Created Date to Closed Date. Note that duration may be censored for
#'   some requests.

dat3 = dat2[dat2.agency .== "NYPD", :]
describe(dat3)[!,[1, end-1]]
filter!(:closed_date => !ismissing, dat3)

# using Dates
dat3.duration .= Dates.value.(dat3.closed_date .- dat3.created_date) ./ 60000
# passmissing(Dates.value)
# disallowmissing!(dat3, :duration)

#' - Visualize the distribution of uncensored duration by weekdays/weekend and
#'   by borough, and test whether the distributions are
#'   the same across weekdays/weekends of their creation and across boroughs.

dat3.weekday .= dayofweek.(dat3.created_date) .<= 5;
duration_weekday = dat3.duration[dat3.weekday];
duration_weekend = dat3.duration[.!dat3.weekday];

#' pick a backend for ploting
#+ eval=false
plt.plotlyjs()    # use plotlyjs backend
plt.plotly()      # use plotly backend
plt.pythonplot()  # use pythonplot backend
plt.gr()          # use gr backend; it's the default

#+ eval=true
sp.@df dat3 sp.density(:duration, group=:weekday, lw=3, label=["weekend" "weekday"])

sp.violin(["duration"], duration_weekday, side=:left, label="weekday")
sp.violin!(["duration"], duration_weekend, side=:right, label="weekend")

sp.@df dat3 sp.boxplot(ifelse.(:weekday, "weekday", "weekend"), :duration,
                       notch=true, label=false)
sp.@df dat3 sp.violin!(ifelse.(:weekday, "weekday", "weekend"), :duration,
                       label=false)

sp.@df dat3 sp.density(:duration, group=:borough, lw=3)
sp.@df dat3 sp.violin(:borough, :duration, label=false)
sp.@df dat3 sp.boxplot(:borough, :duration, label=false)

#' Plot with Makie.jl. Again, you can activate different backends.
# CairoMakie.activate!()
# GLMakie.activate!()

#' Histograms
g = groupby(dat3, :weekday);
b = combine(g, :duration => Ref);

f = Figure()
for i in 1:2
    ax = Axis(f[1, i], title = "Duration on $(i==1 ? "weekend" : "weekday")")
    hist!(ax, b[i,2], normalization=:probability,
          bar_labels=:values, label_formatter=x->round(x, digits=3),
          label_size=15, strokewidth=0.5, strokecolor=(:black, 0.5),
          color=:values)
    ylims!(ax, 0, 0.8)
end
f

#' Density plots
f = Figure()
ax = Axis(f[1, 1], title = "Duration on weekdays and weekends")
d1 = density!(duration_weekday, offset=0, strokewidth=2, strokecolor=:black,
              color=(:red, 0.6))
d2 = density!(duration_weekend, strokewidth=2, strokecolor=:black,
              color=(:green, 0.6))
axislegend(ax, [d1, d2], ["weekday", "weekend"], position=:rc)
f

#' by borough
g = groupby(dat3, :borough);
b = combine(g, :duration => Ref);
f = Figure()
ax = Axis(f[1, 1])
for i in 1:size(b,1)
    density!(ax, b[i,2], strokewidth=2)
end
f

#' Boxplots
f = Figure()
ax = Axis(f[1, 1], xticks = (0:1, ["weekend", "weekday"]), 
          title = "Duration on weekdays and weekends")
d = boxplot!(dat3.weekday, dat3.duration, show_notch=true,
             color=ifelse.(dat3.weekday, :blue, :green), outliercolor=:red)
f

#' HypothesisTests
using HypothesisTests
EqualVarianceTTest(duration_weekday, duration_weekend)
UnequalVarianceTTest(duration_weekday, duration_weekend)
MannWhitneyUTest(duration_weekday, duration_weekend)

g = groupby(dat3, :borough);
b = combine(g, :duration => Ref)
OneWayANOVATest(b.duration_Ref...)
KruskalWallisTest(b.duration_Ref...)

#' - Basic information at the zipcode level such as population density, median
#'   home value, and median household income is available from the US
#'   Census. Convenient accesses are, for example, R package `zipcodeR` and
#'   Python package `uszipcode`; there seems to no Julia equivalent yet but
#'   Julia can call R or Python easily. Merge the zipcode level information
#'   with the NYPD requests data.

dat4 = filter(:latitude => !ismissing, dat3)

#' PythonCall.jl can be use to interact with Python. You can need to use the following code to install Python packages.

#+ eval=false; echo = true; results = "hidden"
using CondaPkg
CondaPkg.add_pip("uszipcode")
CondaPkg.add("python-Levenshtein")

#+ eval=true; echo = true
uszipcode = pyimport("uszipcode")

sr = uszipcode.SearchEngine()
sr.by_zipcode("10001")
sr.by_coordinates(40.8636, -73.8704, returns=1)[0]

#' Using a for loop to find one data point a time in Python from julia is slow. I am sure it can be done much faster. A csv file is created to save time. 
if abspath(PROGRAM_FILE) == @__FILE__ 
    usnames = ("units", "area", "value", "income", "occupiedunits",
               "population", "population_density", "water_area", "zip")
    dus = length(usnames)
    us = DataFrame([name => Int[] for name in usnames])
    @time for r in eachrow(dat4)#[1:10]
        z = sr.by_coordinates(r.latitude, r.longitude, returns=1)
        if isempty(z)
            tmp = fill(missing, dus)
        else
            z = sr.by_coordinates(r.latitude, r.longitude, returns=1)[0]
            tmp = (z.housing_units, z.land_area_in_sqmi, z.median_home_value,
                   z.median_household_income, z.occupied_housing_units,
                   z.population, z.population_density, z.water_area_in_sqmi,
                   z.zipcode)
            tmp = pyconvert(Tuple, tmp)
        end
        push!(us, replace(tmp, nothing => missing), promote=true)
    end
    CSV.write("uszipdata.csv", us)
else
    us = CSV.read("uszipdata.csv", DataFrame)
end

dat5 = [dat4[!, Not(:bbl, :incident_address, :incident_zip)] us]
dropmissing!(dat5)

#' + Data analysis. 
#' - Define a binary variable `over3h` which is 1 if duration is greater than 3
#'   hours. Note that it can be obtained even for censored duration.

dat5.over3h .= dat5.duration .> 180

#' 	- Build a logistic model to predict over3h using the 311 request data as
#'   well as those zip code level covariates. If your model has tuning
#'   parameters, justify their choices. Use appropriate metrics to assess the
#'   performance of the model.

using GLM, StatsBase

fm = @formula(over3h ~ latitude + longitude + weekday + value + income
              + population + water_area)
lgst = glm(fm, dat5, Bernoulli())

p̂ = GLM.predict(lgst)
mean((p̂ .> 0.5) .== dat5.over3h) # accuracy

function cal_auc(p, y)
    p1, p0 = p[y .== 1], p[y .== 0]
    n1, n0 = length(p1), length(p0)
    s = 0.0
    for i in p0, j in p1
        s += (i < j) + 0.5(i == j)
    end
    s / (n1*n0)
end            

@time cal_auc(p̂, dat5.over3h)

using FLoops

function cal_aucF(p, y)
    p1, p0 = p[y .== 1], p[y .== 0]
    n1, n0 = length(p1), length(p0)
    @floop for i in p0, j in p1
        @reduce(s += (i < j) + 0.5(i == j))
    end
    s / (n1*n0)
end            

@time cal_auc(p̂, dat5.over3h)
@time cal_aucF(p̂, dat5.over3h)

using MLJ
accuracy(p̂ .> 0.5, dat5.over3h)

y_cat = categorical(dat5.over3h);
c = categorical(unique(y_cat))
ŷ = [UnivariateFinite(c, [1.0 - p, p]) for p in p̂];
@time auc(ŷ, y_cat)
rc = roc_curve(ŷ, y_cat);
plt.plot(rc[1:2]..., color=:red, lw=3, label="glm", legend=:right,
         xlab="False Positive", ylab="True Positive")

#' - Repeat the analysis with another model (e.g., random forest; neural
#'   network; etc.).
#'

using XGBoost
X = dat5[!,[:latitude, :longitude, :weekday, :value, :income, :population, :water_area]]
Y = dat5.over3h

bst = xgboost((X,Y), num_round=100, max_depth=10, objective="binary:logistic")
p̂_xgb = XGBoost.predict(bst, X)
accuracy(p̂_xgb .> 0.5, y_cat)
cal_auc(p̂_xgb, Y)

ŷ_xgb = [UnivariateFinite(c, [1.0 - p, p]) for p in p̂_xgb];
rc_xgb = roc_curve(ŷ_xgb, y_cat)
plt.plot!(rc_xgb[1:2]..., color=:blue, lw=3, label="xgboost")

#+ eval=false; echo = false; results = "hidden"
using Weave
ENV["GKSwstype"]="nul"
# using ElectronDisplay; ElectronDisplay.CONFIG.focus = false
# get_chunk_defaults()
set_chunk_defaults!(:term => true)
weave("code.jl", doctype="github", out_path="readme.md")
