using LinearAlgebra
using QuantumInformation
using Optim
using Plots

function super_operator(H, Jphi, t)
    """
        Input:
        H : hermitian 2x2 matrix
        Jphi : Choi isomorphism of qubit CP operation Φ, [Jphi ≥ 0]
        t : time, [t ≥ 0]
        Output:
        super_operator : superoperator of the Markovian evolution at time t
                         for Lindblad generator i[⋅, H] + Φ(⋅) - 1/2 {Φ'(1), ⋅}
    """
    K = reshuffle(Jphi)
    X = ptrace(reshuffle(K'), [2,2], 2)
    M = - im * H ⊗ I(2) + im * I(2) ⊗ conj.(H) + K - ( X ⊗ I(2) + I(2) ⊗ conj.(X) ) / 2
    return exp(M*t)
end

function general(a, b, L)
    """
        Input:
        a,b : a,b ∈ [0,1], such that [
                                        a 1-b
                                        1-a b
                                      ]
             is 2x2 stochastic matrix
        L : warm start points for optimization
        Output:
        general : minimized L1 norm of the vector
                  (<1|exp(Lt)(|1><1|)|1> - a, <2|exp(Lt)(|2><2|)|2> - b)
                  over Lt domain
    """
    function f(input)
        t, h = input[1:2]
        G = input[3:34]

        H = [cos(h) sin(h); sin(h) -cos(h)]
        G = reshape(G[1:16], (4,4)) + im * reshape(G[17:32], (4,4))
        Jphi = G * G'

        E = super_operator(H, Jphi, t)
        return maximum([abs(E[1,1] - a), abs(E[4, 4] - b)])
    end

    lower = fill(-Inf, 34)
    upper = fill(Inf, 34)
    lower[1] = 0
    solution = Optim.optimize(f, lower, upper, L, Fminbox(NelderMead()))
    return Optim.minimum(solution)
end

function simple(a, b, L)
    """
        Input:
        a,b : a,b ∈ [0,1], such that [
                                        a 1-b
                                        1-a b
                                      ]
             is 2x2 stochastic matrix
        L : warm start points for optimization
        Output:
        simple : minimized L1 norm of the vector
                 (<1|exp(Lt)(|1><1|)|1> - a, <2|exp(Lt)(|2><2|)|2> - b)
                 over simplified Lt domain
    """

    function f(input)
        t, gamma, alpha, beta = input

        H = sx
        vec_out = [cos(alpha), sin(alpha) * im] 
        vec_in = [cos(beta), sin(beta) * im]
        Jphi = gamma * (vec_out * vec_out') ⊗ conj.(vec_in * vec_in')
        
        E = super_operator(H, Jphi, t)
        return maximum([abs(E[1,1] - a), abs(E[4, 4] - b)])
    end

    lower = fill(-Inf, 4)
    upper = fill(Inf, 4)
    lower[1] = 0
    lower[2] = 0
    solution = Optim.optimize(f, lower, upper, L[1:4], Fminbox(NelderMead()))
    return Optim.minimum(solution)
end

function result(method, N, k, delta = 1e-04)
    """
        Input:
        N: the number of division of the interval [0,1]
        k: the number of used random warm starts for method(a, b, L)
        delta: precision of L1 norm
        Output:
        result : list of pairs (i/N, b_i), such that
                 method(i/N, b_i, rand(34)) < delta, but
                 method(i/N, b_i - delta, random) ≥ delta

    """
    L = []
    for a in (0:N)/N
        bd, bu = 0, a
        while bu - bd > delta
            b = (bd + bu) / 2
            temp_dist = 2
            for _ in 1:k
                result = method(a, b, rand(34))
                if result < temp_dist
                    temp_dist = result
                end
            end
            if temp_dist < delta
                bu = b
            else
                bd = b
            end
        end
        append!(L, [(a, bu)])
    end
    return L
end

# GENERATE RESULTS IN GENERAL CASE
k = 10
N = 100
list_result_general = result(general, N, k)
image_result_general = plot([a[1] for a in list_result_general], 
[b[2] for b in list_result_general], color = "blue", lw = 1)

# GENERATE RESULTS IN SIMPLIFIED CASE & COMPARE WITH GENERAL CASE
list_result_simple = result(simple, N, k)
image_result = plot!(image_result_general, [a[1] for a in list_result_simple], 
[b[2] for b in list_result_simple], color = "red", lw = 0.5)

savefig(image_result, "results.pdf")
print(minimum([list_result_general[i][2] - list_result_simple[i][2] for i=1:(N+1)]))

