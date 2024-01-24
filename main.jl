include("autoEncoder.jl")
include("regression.jl")

function main()
    data = L.get_data(1)[1].data
    autoEncoder = L.load_model("autoEncoder.jld2")
    regression = P.load_model("regression.jld2")
    latent = autoEncoder[2](autoEncoder[1](data))
    reg = regression(latent)
    predict = autoEncoder[4](autoEncoder[3](reg))

    # plot(vcat(collect(p[:,i] for i = 1:104)...)) # plot the entire timeline

    return data, predict
end