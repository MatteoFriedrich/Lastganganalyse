module P

using CSV
using DataFrames
using Flux
using Flux.Data: DataLoader
using GLMakie
using JLD2
using StatsBase


function get_data()
    df = CSV.read("latent.csv", DataFrame)
    latent = zeros(3,90)
    for i = 1:90
        latent[:,i] = df[:, "x"*string(i)]
    end

    xtrain = latent[:,1:89]
    ytrain = latent[:,2:90]
    xtest = zeros(3,10)
    ytest = zeros(3,10)

    test_indexes = sample(1:89, 10, replace=false)
    xtest = zeros(3,10)
    ytest = zeros(3,10)
    for i in eachindex(test_indexes)
        xtest[:,i] = xtrain[:,test_indexes[i]]
        ytest[:,i] = ytrain[:,test_indexes[i]]
    end

    xtrain = xtrain[:, setdiff(1:end, test_indexes)]
    ytrain = ytrain[:, setdiff(1:end, test_indexes)]

    return (Flux.DataLoader((data=xtrain, label=ytrain), batchsize=5, shuffle=true), Flux.DataLoader((data=xtest, label=ytest)))
    #return ((xtrain,ytrain), (xtest,ytest))
end

function save_model(path, model)
    jldsave(path; model)
end

function load_model(path)
    return JLD2.load(path, "model")
end

function train!(model_loss, model_params, opt, train_data, test_data, device, epochs = 10)
    train_steps = 0
    "Start training for total $(epochs) epochs" |> println
    for epoch = 1:epochs
        
        ℒ = 0
        for (x,y) in train_data
            loss, back = Flux.pullback(model_params) do
                model_loss(x |> device, y |> device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, model_params, grad)
            train_steps += 1
            ℒ += loss
        end
        if epoch % 1000 == 0
            print("Epoch $(epoch): ")
            println("ℒ = $ℒ")
            println("test_loss", model_loss(test_data.data[1], test_data.data[2]))
        end
    end
    "Total train steps: $train_steps" |> println
end

function main(new_model = true)
    device = cpu # where will the calculations be performed?
    η = 0.00001 # learning rate for ADAM optimization algorithm


    train_data, test_data = get_data()
    if new_model
        m = Chain(Dense(3,3))
    else
        m = load_model("regression.jld2")
    end

    loss(x,y) = Flux.Losses.mse(m(x), y)

    #loss(data_sample)

    opt = ADAM(η)
    ps = Flux.params(m) # parameters
    train!(loss, ps, opt, train_data, test_data,device, 20000)

    save_model("regression.jld2", m)

    return train_data, test_data, m
end

end # module