#!/usr/bin/env julia
# scripts/julia_preprocess.jl
#
# Usage:
#   julia --project=. scripts/julia_preprocess.jl --config config/pre_process.yaml

using YAML
using HDF5
using Random

const MINKOWSKI_DIAG = [-1.0, 1.0, 1.0, 1.0]  # diag(-1, +1, +1, +1)

# Simple CLI parser: we only care about --config <path>
function parse_args()
    config_path = "config/pre_process.yaml"
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--config"
            i == length(ARGS) && error("--config expects a path")
            config_path = ARGS[i + 1]
            i += 1
        else
            @warn "Ignoring unknown argument $arg"
        end
        i += 1
    end
    return config_path
end

function load_config(path::AbstractString)
    println("Loading config from $path")
    return YAML.load_file(path)
end

# Minkowski dot product with signature (-,+,+,+)
function dot_four(a::AbstractVector, b::AbstractVector)
    @assert length(a) == 4 "four-vectors must have length 4"
    @assert length(b) == 4 "four-vectors must have length 4"
    s = 0.0
    @inbounds for μ in 1:4
        s += MINKOWSKI_DIAG[μ] * a[μ] * b[μ]
    end
    return s
end

"""
Compute the 27 Lorentz-invariant scalars per sample as in Eq. (2) of the paper:
21 pairwise four-flux dot products + 6 flux–velocity products, each normalized
by N_tot = -∑_species F⋅u.
F has shape (N, 6, 4), u has shape (N, 4).
Returns an array of shape (N, 27).
"""
function compute_invariants(F::Array{<:Real,3}, u::Array{<:Real,2})
    N, nspecies, ncomp = size(F)
    @assert nspecies == 6 "expected 6 neutrino species (νe, νμ, ντ, ν̄e, ν̄μ, ν̄τ)"
    @assert ncomp == 4 "four-flux must have 4 components"
    @assert size(u, 1) == N && size(u, 2) == 4 "u must have shape (N, 4)"

    invariants = Array{Float32}(undef, N, 27)

    for k in 1:N
        uk = @view u[k, :]
        # views for each species
        Fk = [@view F[k, a, :] for a in 1:nspecies]

        # F⋅u for each species and N_tot
        Fdotu = zeros(Float64, nspecies)
        N_tot = 0.0
        for a in 1:nspecies
            Fdotu[a] = dot_four(Fk[a], uk)
            N_tot -= Fdotu[a]  # N_tot = -∑ F⋅u  (should be > 0)
        end

        den = N_tot ≈ 0 ? 1e-30 : N_tot

        idx = 1
        # pairwise F-F dot products (a ≤ b)
        for a in 1:nspecies
            for b in a:nspecies
                sab = dot_four(Fk[a], Fk[b]) / den
                invariants[k, idx] = sab
                idx += 1
            end
        end

        # F-u dot products
        for a in 1:nspecies
            invariants[k, 21 + a] = Fdotu[a] / den
        end
    end

    return invariants
end

"""
Compute the residual target ΔF = F_true - F_box and flatten to (N, 24)
by stacking 6 species × 4 components, as in the paper.
"""
function compute_residual(F_true::Array{<:Real,3}, F_box::Array{<:Real,3})
    @assert size(F_true) == size(F_box) "F_true and F_box must have same shape"
    N, nspecies, ncomp = size(F_true)
    @assert nspecies == 6 "expected 6 neutrino species"
    @assert ncomp == 4 "four-flux must have 4 components"

    residual = Array{Float32}(undef, N, nspecies * ncomp)

    for k in 1:N
        idx = 1
        @inbounds for a in 1:nspecies
            for μ in 1:ncomp
                residual[k, idx] = Float32(F_true[k, a, μ] - F_box[k, a, μ])
                idx += 1
            end
        end
    end

    return residual
end

"""
Process a single HDF5 file according to the config entry.
Returns a NamedTuple with all arrays for that file.
"""
function process_file(cfg, file_cfg)
    data_root = get(cfg, "data_root", "data")
    h5_paths = cfg["h5_paths"]

    rel_path = file_cfg["path"]
    sim_path = joinpath(data_root, rel_path)
    sim_id = String(file_cfg["sim_id"])
    max_rows = get(file_cfg, "max_rows", nothing)

    println("------------------------------------------------------------")
    println("Processing file: $sim_path")
    println("  sim_id = $sim_id")
    println("  max_rows = $(max_rows === nothing ? "all" : string(max_rows))")

    F_init = nothing
    F_true = nothing
    F_box = nothing
    u = nothing

    h5open(sim_path, "r") do h5
        F_init = read(h5[h5_paths["F_init"]])
        F_true = read(h5[h5_paths["F_true"]])
        F_box  = read(h5[h5_paths["F_box"]])
        u      = read(h5[h5_paths["u"]])
    end

    N = size(F_init, 1)
    @assert size(F_true, 1) == N && size(F_box, 1) == N && size(u, 1) == N

    if max_rows !== nothing && max_rows > 0 && max_rows < N
        N = max_rows
        F_init = F_init[1:N, :, :]
        F_true = F_true[1:N, :, :]
        F_box  = F_box[1:N, :, :]
        u      = u[1:N, :]
        println("  truncated to first $N rows")
    else
        println("  using all $N rows")
    end

    # compute targets & features
    residual   = compute_residual(F_true, F_box)
    invariants = compute_invariants(F_init, u)  # invariants from initial four-flux

    sim_ids = fill(sim_id, N)

    # convert to Float32 for storage
    F_init32 = Float32.(F_init)
    F_true32 = Float32.(F_true)
    F_box32  = Float32.(F_box)

    return (F_init = F_init32,
            F_true = F_true32,
            F_box  = F_box32,
            residual = residual,
            invariants = invariants,
            sim_id = sim_ids)
end

function write_sim_file(output_root::AbstractString,
                        file_cfg::Dict{String,Any},
                        data)
    mkpath(output_root)
    name = get(file_cfg, "output_name", string(file_cfg["name"], "_preprocessed.h5"))
    out_path = joinpath(output_root, name)

    println("  writing per-simulation file: $out_path")

    h5open(out_path, "w") do h5
        write(h5, "F_init", data.F_init)
        write(h5, "F_true", data.F_true)
        write(h5, "F_box", data.F_box)
        write(h5, "residual", data.residual)
        write(h5, "invariants", data.invariants)
        write(h5, "sim_id", data.sim_id)
    end
end

function main()
    config_path = parse_args()
    cfg = load_config(config_path)

    output_root = get(cfg, "output_root", "pdata")
    output_filename = get(cfg, "output_filename", "preprocessed_all.h5")
    shuffle = get(cfg, "shuffle", true)
    seed    = get(cfg, "seed", 42)

    files_cfg = cfg["files"]
    @assert !isempty(files_cfg) "config.files is empty"

    Random.seed!(seed)

    datasets = Vector{Any}()

    for file_cfg_any in files_cfg
        file_cfg = Dict{String,Any}(file_cfg_any)
        enabled = get(file_cfg, "enabled", true)
        if !enabled
            println("Skipping disabled file $(file_cfg["name"])")
            continue
        end

        data = process_file(cfg, file_cfg)
        write_sim_file(output_root, file_cfg, data)
        push!(datasets, data)
    end

    isempty(datasets) && error("No enabled files were processed; nothing to write.")

    # Concatenate all datasets
    F_init_all     = cat((d.F_init for d in datasets)...; dims=1)
    F_true_all     = cat((d.F_true for d in datasets)...; dims=1)
    F_box_all      = cat((d.F_box  for d in datasets)...; dims=1)
    residual_all   = cat((d.residual for d in datasets)...; dims=1)
    invariants_all = cat((d.invariants for d in datasets)...; dims=1)
    sim_id_all     = vcat((d.sim_id for d in datasets)...)

    N_total = size(F_init_all, 1)
    println("------------------------------------------------------------")
    println("Assembled combined dataset with $N_total samples")

    if shuffle
        println("Shuffling combined dataset with seed = $seed")
        perm = randperm(N_total)
        F_init_all     = F_init_all[perm, :, :]
        F_true_all     = F_true_all[perm, :, :]
        F_box_all      = F_box_all[perm, :, :]
        residual_all   = residual_all[perm, :]
        invariants_all = invariants_all[perm, :]
        sim_id_all     = sim_id_all[perm]
    else
        println("Shuffling disabled")
    end

    mkpath(output_root)
    out_path = joinpath(output_root, output_filename)

    println("Writing combined dataset to $out_path")

    h5open(out_path, "w") do h5
        write(h5, "F_init", F_init_all)
        write(h5, "F_true", F_true_all)
        write(h5, "F_box",  F_box_all)
        write(h5, "residual", residual_all)
        write(h5, "invariants", invariants_all)
        write(h5, "sim_id", sim_id_all)
    end

    println("Done.")
end

main()
