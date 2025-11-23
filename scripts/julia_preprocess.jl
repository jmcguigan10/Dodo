#!/usr/bin/env julia
# scripts/julia_preprocess.jl
#
# Preprocessing Emu_data (F4_initial, F4_final, directorynames) into:
#   - F_init:     (N, 6, 4)  initial four-fluxes
#   - F_true:     (N, 6, 4)  final four-fluxes (Emu)
#   - F_box:      (N, 6, 4)  baseline four-fluxes (currently just F_init)
#   - residual:   (N, 24)    flattened ΔF = F_true - F_box
#   - invariants: (N, 27)    Lorentz scalars from initial F
#   - sim_id:     (N,)       coarse simulation ID (3ms/7ms/2016/random)
#   - dirname:    (N,)       original Emu directorynames (optional)
#
# Config is read from YAML; see config/pre_process.yaml.

using YAML
using HDF5
using Random
include("./Box3D.jl")
include("./get_cfg.jl")
# Metric diag(1,1,1,-1) with components ordered (x, y, z, t)
const MINKOWSKI_DIAG = [1.0, 1.0, 1.0, -1.0]

# Fluid 4-velocity in Emu_data: comoving frame, (x,y,z,t) ordering
const U_VEC = [0.0, 0.0, 0.0, 1.0]

# ---------------------------------------------------------------------------

# Minkowski dot product with signature (+,+,+,-) in (x,y,z,t)
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
Convert F4 from Emu_data format into (N, 6, 4):

We only assume that dims 2–4 are some permutation of sizes (4, 2, 3)
corresponding to:
  4  -> spacetime components (x,y,z,t)
  2  -> neutrino / antineutrino
  3  -> flavor (e, mu, tau)

This auto-detects the permutation and permutes to (N, 4, 2, 3) before
flattening (2,3) -> 6 species.
"""
function convert_F4_to_species(F4::Array{<:Real,4})
    dims = size(F4)

    # Find which dim is which
    idx_comp = findfirst(==(4), dims)   # spacetime components
    idx_nnu  = findfirst(==(2), dims)   # nu / antinu
    idx_flav = findfirst(==(3), dims)   # flavor
    idx_species = findfirst(==(6), dims)

    function sample_idx(used)
        used_vec = collect(used)
        idxs = setdiff(collect(1:length(dims)), used_vec)
        length(idxs) == 1 || error("could not infer sample dimension from shape $(dims)")
        return first(idxs)
    end

    # Some datasets combine flavor and nu/antinu into a single species
    # dimension of size 6. Detect that case and reshape directly.
    if (idx_flav === nothing || idx_nnu === nothing) && idx_comp !== nothing && idx_species !== nothing
        idx_sample = sample_idx((idx_comp, idx_species))
        perm = (idx_sample, idx_comp, idx_species)
        F4p = perm == (1, 2, 3) ? F4 : permutedims(F4, perm)

        N, ncomp, nspecies = size(F4p)
        @assert ncomp == 4 "post-permute: expected 4 spacetime components, got $ncomp"
        @assert nspecies == 6 "post-permute: expected 6 species, got $nspecies"

        F = Array{Float32}(undef, N, 6, 4)
        @inbounds for k in 1:N, a in 1:6, μ in 1:4
            F[k, a, μ] = Float32(F4p[k, μ, a])
        end

        return F
    end

    @assert idx_comp !== nothing "could not find dim of size 4 (spacetime components) in F4"
    @assert idx_nnu  !== nothing "could not find dim of size 2 (nu/antinu) in F4"
    @assert idx_flav !== nothing "could not find dim of size 3 (flavor) in F4; if your data already combines flavor, store it as a size-6 species dimension instead."

    idx_sample = sample_idx((idx_comp, idx_nnu, idx_flav))

    # Permute so dims become: (N, 4, 2, 3)
    perm = (idx_sample, idx_comp, idx_nnu, idx_flav)
    F4p = perm == (1, 2, 3, 4) ? F4 : permutedims(F4, perm)

    N, ncomp, nnu, nflav = size(F4p)
    @assert ncomp == 4 "post-permute: expected 4 spacetime components, got $ncomp"
    @assert nnu  == 2 "post-permute: expected 2 neutrino/antineutrino, got $nnu"
    #@assert nflav == 3 "post-permute: expected 3 flavors, got $nflav"

    # Now F4p[k, μ, inu, f] with μ in {1..4}, inu in {1..2}, f in {1..3}
    # Flatten (inu, f) -> species index a = 1..6
    F = Array{Float32}(undef, N, 6, 4)

    @inbounds for k in 1:N
        for inu in 1:2              # 1=nu, 2=antinu
            for f in 1:3            # 1=e, 2=mu, 3=tau
                a = (inu - 1) * 3 + f
                for μ in 1:4        # μ: 1=x,2=y,3=z,4=t
                    F[k, a, μ] = Float32(F4p[k, μ, inu, f])
                end
            end
        end
    end

    return F
end

# Map 6-species fluxes (x,y,z,t ordering) to the 4-species layout
# (t,x,y,z) × (νe, νebar, νx, νxbar) required by Box3D.
function add_species!(Fbox::AbstractMatrix, col::Int, Fspec)
    Fbox[1, col] += Fspec[4]  # t
    Fbox[2, col] += Fspec[1]  # x
    Fbox[3, col] += Fspec[2]  # y
    Fbox[4, col] += Fspec[3]  # z
end

function set_species!(F6::Array{Float32,3}, k::Int, a::Int, col::Int,
                      Fbox::AbstractMatrix, scale::Real=1.0)
    F6[k, a, 1] = Float32(scale * Fbox[2, col])  # x
    F6[k, a, 2] = Float32(scale * Fbox[3, col])  # y
    F6[k, a, 3] = Float32(scale * Fbox[4, col])  # z
    F6[k, a, 4] = Float32(scale * Fbox[1, col])  # t
end

# Apply Box3D to six-species fluxes by combining μ/τ into νx/νxbar,
# running the 4-species solver, then splitting back evenly into μ/τ.
function box3d_flux_on_six(F6::Array{<:Real,3}; mu_split::Real=0.5)
    N, nspecies, ncomp = size(F6)
    @assert nspecies == 6 "expected 6 species for Box3D mapping"
    @assert ncomp == 4 "expected 4 components (x,y,z,t)"

    μ_frac = clamp(mu_split, 0.0, 1.0)
    τ_frac = 1.0 - μ_frac

    F_box = Array{Float32}(undef, N, 6, 4)
    F_box4 = zeros(Float64, 4, 4)

    for k in 1:N
        fill!(F_box4, 0.0)

        add_species!(F_box4, 1, @view F6[k, 1, :])  # νe
        add_species!(F_box4, 2, @view F6[k, 4, :])  # ν̄e
        add_species!(F_box4, 3, @view F6[k, 2, :])  # νμ
        add_species!(F_box4, 3, @view F6[k, 3, :])  # ντ
        add_species!(F_box4, 4, @view F6[k, 5, :])  # ν̄μ
        add_species!(F_box4, 4, @view F6[k, 6, :])  # ν̄τ

        F_box4_out = box3d_flux(F_box4)

        # Map back to 6 species, splitting νx/ν̄x evenly into μ/τ.
        set_species!(F_box, k, 1, 1, F_box4_out)
        set_species!(F_box, k, 4, 2, F_box4_out)
        set_species!(F_box, k, 2, 3, F_box4_out, μ_frac)
        set_species!(F_box, k, 3, 3, F_box4_out, τ_frac)
        set_species!(F_box, k, 5, 4, F_box4_out, μ_frac)
        set_species!(F_box, k, 6, 4, F_box4_out, τ_frac)
    end

    return F_box
end


"""
Compute the 27 scalar invariants per sample:

  * 21 pairwise S_ab = (F_a · F_b) / N_tot for a ≤ b
  *  6 s_a   = (F_a · u) / N_tot

with N_tot = -∑_a F_a · u = ∑_a F_a^t > 0,
using metric diag(1,1,1,-1) and u = (0,0,0,1) in (x,y,z,t).
F has shape (N, 6, 4).
"""
function compute_invariants(F::Array{<:Real,3})
    N, nspecies, ncomp = size(F)
    @assert nspecies == 6 "expected 6 neutrino species"
    @assert ncomp == 4 "four-flux must have 4 components (x,y,z,t)"

    invariants = Array{Float32}(undef, N, 27)

    for k in 1:N
        Fk = [@view F[k, a, :] for a in 1:nspecies]

        Fdotu = zeros(Float64, nspecies)
        N_tot = 0.0

        @inbounds for a in 1:nspecies
            Fdotu[a] = dot_four(Fk[a], U_VEC)  # = -F_a^t
            N_tot -= Fdotu[a]                  # N_tot = -∑ F·u = ∑ F^t
        end

        den = N_tot ≈ 0 ? 1e-30 : N_tot

        idx = 1
        @inbounds for a in 1:nspecies
            for b in a:nspecies
                sab = dot_four(Fk[a], Fk[b]) / den
                invariants[k, idx] = Float32(sab)
                idx += 1
            end
        end

        @inbounds for a in 1:nspecies
            invariants[k, 21 + a] = Float32(Fdotu[a] / den)
        end
    end

    return invariants
end

"""
Compute residual ΔF = F_true - F_box and flatten
6 species × 4 components -> (N, 24).
"""
function compute_residual(F_true::Array{<:Real,3}, F_box::Array{<:Real,3})
    @assert size(F_true) == size(F_box) "F_true and F_box must have same shape"
    N, nspecies, ncomp = size(F_true)
    @assert nspecies == 6 "expected 6 species"
    @assert ncomp == 4 "expected 4 components"

    residual = Array{Float32}(undef, N, nspecies * ncomp)

    @inbounds for k in 1:N
        idx = 1
        for a in 1:nspecies
            for μ in 1:ncomp
                residual[k, idx] = Float32(F_true[k, a, μ] - F_box[k, a, μ])
                idx += 1
            end
        end
    end

    return residual
end

"""
Process a single Emu_data HDF5 file according to config entry.
Returns a NamedTuple with all arrays for that file.
"""
function process_file(cfg, file_cfg::Dict{String,Any}, mu_split::Real)
    data_root = get(cfg, "data_root", "data")
    h5_paths = cfg["h5_paths"]

    rel_path = file_cfg["path"]
    sim_path = joinpath(data_root, rel_path)
    sim_id = String(file_cfg["sim_id"])
    max_rows = get(file_cfg, "max_rows", nothing)

    println("------------------------------------------------------------")
    println("Processing file: $sim_path")
    println("  sim_id   = $sim_id")
    println("  max_rows = $(max_rows === nothing ? "all" : string(max_rows))")

    F4_init = nothing
    F4_final = nothing
    dirnames = nothing

    h5open(sim_path, "r") do h5
        F4_init  = read(h5[h5_paths["F_init"]])
        F4_final = read(h5[h5_paths["F_true"]])

        if haskey(h5_paths, "dirnames") && haskey(h5, h5_paths["dirnames"])
            dirnames = read(h5[h5_paths["dirnames"]])
        end
    end

    # Convert to (N, 6, 4)
    F_init6 = convert_F4_to_species(F4_init)
    F_true6 = convert_F4_to_species(F4_final)
    N_total = size(F_init6, 1)
    @assert size(F_true6, 1) == N_total "F4_initial and F4_final must have same number of samples"

    if dirnames !== nothing
        dirnames = vec(String.(dirnames))
    end

    if max_rows !== nothing && max_rows > 0 && max_rows < N_total
        N = max_rows
        F_init6 = F_init6[1:N, :, :]
        F_true6 = F_true6[1:N, :, :]
        if dirnames !== nothing && length(dirnames) >= N
            dirnames = dirnames[1:N]
        end
        println("  truncated to first $N rows")
    else
        N = N_total
        if dirnames !== nothing && length(dirnames) != N
            dirnames = length(dirnames) >= N ? dirnames[1:N] : fill("", N)
        end
        println("  using all $N rows")
    end

    # Use Box3D flux function to compute predicted final flux
    F_box6 = box3d_flux_on_six(F_init6; mu_split=mu_split)

    residual   = compute_residual(F_true6, F_box6)
    invariants = compute_invariants(F_init6)

    sim_ids = fill(sim_id, N)

    # directorynames: keep for debugging, but not required for training
    dirname_vec =
        dirnames === nothing ? fill("", N) : String.(dirnames)

    F_init32 = Float32.(F_init6)
    F_true32 = Float32.(F_true6)
    F_box32  = Float32.(F_box6)

    return (F_init = F_init32,
            F_true = F_true32,
            F_box = F_box32,
            residual = residual,
            invariants = invariants,
            sim_id = sim_ids,
            dirname = dirname_vec)
end

function write_sim_file(output_root::AbstractString,
                        file_cfg::Dict{String,Any},
                        data)
    mkpath(output_root)
    name = get(file_cfg, "output_name", string(file_cfg["name"], "_preprocessed.h5"))
    out_path = joinpath(output_root, name)

    println("  writing per-simulation file: $out_path")

    h5open(out_path, "w") do h5
        write(h5, "F_init",     data.F_init)
        write(h5, "F_true",     data.F_true)
        write(h5, "F_box",      data.F_box)
        write(h5, "residual",   data.residual)
        write(h5, "invariants", data.invariants)
        write(h5, "sim_id",     data.sim_id)
        write(h5, "dirname",    data.dirname)
    end
end

function main()
    config_path = parse_args()
    cfg = load_config(config_path)

    output_root     = get(cfg, "output_root", "pdata")
    output_filename = get(cfg, "output_filename", "preprocessed_all.h5")
    mu_split        = get(cfg, "box3d_mu_fraction", 0.5)
    shuffle         = get(cfg, "shuffle", true)
    seed            = get(cfg, "seed", 42)

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

        data = process_file(cfg, file_cfg, mu_split)
        write_sim_file(output_root, file_cfg, data)
        push!(datasets, data)
    end

    isempty(datasets) && error("No enabled files were processed; nothing to write.")

    F_init_all     = cat((d.F_init     for d in datasets)...; dims = 1)
    F_true_all     = cat((d.F_true     for d in datasets)...; dims = 1)
    F_box_all      = cat((d.F_box      for d in datasets)...; dims = 1)
    residual_all   = cat((d.residual   for d in datasets)...; dims = 1)
    invariants_all = cat((d.invariants for d in datasets)...; dims = 1)
    sim_id_all     = vcat((d.sim_id    for d in datasets)...)
    dirname_all    = vcat((d.dirname   for d in datasets)...)

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
        dirname_all    = dirname_all[perm]
    else
        println("Shuffling disabled")
    end

    mkpath(output_root)
    out_path = joinpath(output_root, output_filename)

    println("Writing combined dataset to $out_path")

    h5open(out_path, "w") do h5
        write(h5, "F_init",     F_init_all)
        write(h5, "F_true",     F_true_all)
        write(h5, "F_box",      F_box_all)
        write(h5, "residual",   residual_all)
        write(h5, "invariants", invariants_all)
        write(h5, "sim_id",     sim_id_all)
        write(h5, "dirname",    dirname_all)
    end

    println("Done.")
end

main()
