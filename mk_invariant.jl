using LinearAlgebra

"""
    compute_invariants(F_init, u4; metric=M)

Compute the 27 invariants from:

  - 21 F·F combinations for 6 four-vectors F_a (rows of F_init)
  - 6 F·u combinations with a four-velocity u4

Arguments:
  * `F_init` : 6×4 matrix, each row is a four-vector (F^0, F^1, F^2, F^3)
  * `u4`     : length-4 vector (u^0, u^1, u^2, u^3)
  * `metric` : 4×4 metric tensor (defaults to Minkowski diag(-1,1,1,1))

Returns:
  * `Vector{Float64}` of length 27, in the same order as your Python code:
      F_0·F_0, F_0·F_1, ..., F_0·F_5,
      F_1·F_1, F_1·F_2, ..., F_1·F_5,
      ...
      F_5·F_5,
      F_0·u, ..., F_5·u
"""
function compute_invariants(
    F_init::AbstractMatrix{<:Real},
    u4::AbstractVector{<:Real};
    metric::AbstractMatrix{<:Real} = [-1.0 0.0 0.0 0.0;
                                       0.0 1.0 0.0 0.0;
                                       0.0 0.0 1.0 0.0;
                                       0.0 0.0 0.0 1.0],
)
    @assert size(F_init, 1) == 6 "F_init must be 6×4."
    @assert size(F_init, 2) == 4 "F_init must be 6×4."
    @assert length(u4) == 4 "u4 must have length 4."
    @assert size(metric) == (4, 4) "metric must be 4×4."

    inv = Vector{Float64}(undef, 27)
    idx = 1

    # 21 F·F invariants
    @views begin
        for a in 1:6
            Fa = F_init[a, :]  # 4-vector
            for b in a:6
                Fb = F_init[b, :]
                # Fa·Fb with metric: Fa_μ g^{μν} Fb_ν
                inv[idx] = dot(Fa, metric * Fb)
                idx += 1
            end
        end
    end

    # 6 F·u invariants
    u_g = metric * u4  # lower index with metric
    @views begin
        for a in 1:6
            Fa = F_init[a, :]
            inv[idx] = dot(Fa, u_g)
            idx += 1
        end
    end

    return inv  # Vector{Float64}, length 27
end

