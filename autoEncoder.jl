module L

using CSV
using DataFrames
using Dates
using Flux
using Flux.Data: DataLoader
using GLMakie
using Clustering
using JLD2

function plot_day(day)
    data = parse.(Float64, replace.(day[:, :mw], ","=>"."))
    #println(data)
    Plots.plot(data)
end

function df2float(df, key)
    return parse.(Float64, replace.(df[:, key], ","=>"."))
end

function predict(date)
    data = grouped_days[dayofweek(date), :,:]
    sum = zeros(size(data)[2])
    for i = 1:size(data)[1]
        sum += data[i, :]
    end
    return sum./size(data)[1]
end

function organize_data()
    df = CSV.read("data.csv", DataFrame)
    days = [df[(i-1)*96+1:i*96,:] for i = 1:105]
    dates = []
    date_str = split(days[1][!, :ts][1], " ")[1]
    dates = [Date(parse(Int64, split(split(days[i][!, :ts][1], " ")[1], ".")[3]), 
        parse(Int64, split(split(days[i][!, :ts][1], " ")[1], ".")[2]), 
        parse(Int64, split(split(days[i][!, :ts][1], " ")[1], ".")[1])) for i = eachindex(days)]

    weekdays = [dayofweek(date) for date in dates]
    grouped_days = zeros(Float64, 7, Int(ceil(length(weekdays)/7)), 24*4)
    for i = eachindex(weekdays)
        grouped_days[weekdays[i],Int(ceil(i/7)),:] = df2float(days[i], :mw)
    end
end

function get_data(batch_size)
    df = CSV.read("data.csv", DataFrame)
    days = [df[(i-1)*96+1:i*96,:] for i = 1:105]
    a = reduce(vcat,transpose.(df2float.(days[1:105], :mw)))
    a = rotr90(a)./1000

    dl = DataLoader(a, batchsize=batch_size, shuffle=true)
    dl, 96
end

function save_model(path, model)
    jldsave(path; model)
end

function load_model(path)
    return JLD2.load(path, "model")
end

function train!(model_loss, model_params, opt, loader, device, epochs = 10)
    train_steps = 0
    "Start training for total $(epochs) epochs" |> println
    for epoch = 1:epochs
        
        ℒ = 0
        for x in loader
            loss, back = Flux.pullback(model_params) do
                model_loss(x |> device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, model_params, grad)
            train_steps += 1
            ℒ += loss
        end
        if epoch % 1000 == 0
            	print("Epoch $(epoch): ")
            println("ℒ = $ℒ")
        end
    end
    "Total train steps: $train_steps" |> println
end

function main()
    device = cpu # where will the calculations be performed?
    L1, L2 = 48, 3 # layer dimensions
    η = 0.00001 # learning rate for ADAM optimization algorithm
    batch_size = 100; # batch size for optimization

    dl, d = get_data(90)

    enc1 = Dense(d, L1, leakyrelu)
    enc2 = Dense(L1, L2, sigmoid)
    dec3 = Dense(L2, L1, leakyrelu)
    dec4 = Dense(L1, d)
    m = Chain(enc1, enc2, dec3, dec4)

    loss(x) = Flux.Losses.mse(m(x), x)
    #loss(data_sample)

    opt = ADAM(η)
    ps = Flux.params(m) # parameters
    train!(loss, ps, opt, dl, device, 200000)
    save_model("autoEncoder.jld2", m)
    return dl,m
end

function cluster(dl,m)
    latent = m[2](m[1](dl.data))
    R = kmeans(latent, 4; maxiter=200, display=:iter)

    clusters = [[] for i = 1:5]
    a = assignments(R) # get the assignments of points to clusters
    for i = 1:90
        aa = a[i]
        push!(clusters[aa], latent[:,i])
    end
    fig = GLMakie.Figure()
    ax = GLMakie.Axis3(fig[1, 1])
    
    for i = 1:4
        p = zeros(3,length(clusters[i]))
        for c = 1:length(clusters[i])
            p[:,c] = clusters[i][c]
        end
        scatter!(p)
    end
    display(fig)
end

end