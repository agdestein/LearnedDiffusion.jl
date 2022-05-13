# # Operator inference
#
# In this example, we will identify an unknown operator describing a physical system by
# combining tools from scientific computing (PDEs, discretizations) and machine learning
# (minimizing prediction errors, backpropagation).
#
# For this, the [Julia](https://julialang.org/) language is a natural choice,
# in this example because of automatic differentiation (AD) of native code.

#nb # ## The Julia bootstrap block (for Google Colab)
#nb # Source: https://colab.research.google.com/drive/1_4Yz3FKO5_uuYvamEfHqwtFT9WpCuSbm
#nb #
#nb # This should be run for the first time to install Julia kernel, and then refresh this page (e.g., Ctrl-R)
#nb # so that colab will redirect to the installed Julia kernel
#nb # and then doing your own work

#nb ## 1. install latest Julia using jill.py
#nb ##    tip: one can install specific Julia version using e.g., `jill install 1.7`
#nb !pip install jill && jill install --upstream Official --confirm
#nb 
#nb ## 2. install IJulia kernel
#nb ! julia -e 'using Pkg; pkg"add IJulia"; using IJulia; installkernel("Julia")'
#nb 
#nb ## 3. hot-fix patch to strip the version suffix of the installed kernel so
#nb ## that this notebook kernelspec is version agnostic
#nb !jupyter kernelspec install $(jupyter kernelspec list | grep julia | tr -s ' ' | cut -d' ' -f3) --replace --name julia

#nb #-

#nb using Pkg; Pkg.add(["OrdinaryDiffEq", "DiffEqFlux", "Plots"])

# We start by loading some packages.

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using DiffEqFlux
using Plots

# That may have taken some time. Julia is a compiled language, and she won't
# hesitate to compile any code she can put her eyes on, even just to get a
# small specialization improvement. Once heavy simulations are launched,
# however, this efficiency will be worth your patience. The first time you run
# some of the cells below it may also take some time, as the functions, say,
# `heatmap`, specializes on your input types. Once a function is compiled
# however, it should stay so for the remainder of the session.

# Since the problems will consist of inferring various matrices, we define a small helper
# function for visualizing our results:

plotmat(A; kwargs...) = heatmap(
    reverse(A; dims = 1);
    aspect_ratio = :equal,
    xlims = (1 / 2, size(A, 2) + 1 / 2),
    ylims = (1 / 2, size(A, 1) + 1 / 2),
    ## xticks = nothing,
    ## yticks = nothing,
    kwargs...,
)

# We also provide a specialized method for sparse matrices, which must be densified before
# plotting.

plotmat(A::AbstractSparseMatrix; kwargs...) = plotmat(Matrix(A); kwargs...)


# ## Problem statement
#
# Consider a linear ordinary differential equation (ODE) parameterized by some operator
# $\mathbf{A}$:
#
# $$\frac{\mathrm{d} \mathbf{u}}{\mathrm{d} t} = \mathbf{A} \mathbf{u}, \quad \mathbf{u}(0)
# = \mathbf{u}_0$$
#
# where $\mathbf{u}_0$ are some initial conditions. To solve this system, we will use the
# [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) package. It provides
# differentiable ODE solvers for problems defined by a parametrized ODE function $f$
# defining the right hand side of the ODE for a given state $\mathbf{u}$, time $t$ and
# parameters $p$ (in our case: $p = \mathbf{A}$), i.e. $\frac{\mathrm{d}
# \mathbf{u}}{\mathrm{d} t} = f(\mathbf{u}, p, t)$.

f(u, A, t) = A * u

# For convenience, we will define a solver function $\mathbf{S}: (\mathbf{A}, \mathbf{u}_0,
# t) \mapsto \mathbf{u}(t)$, where $\mathbf{u}$ is the solution to the above system for a
# given operator $\mathbf{A}$ and initial conditions $\mathbf{u}_0$. The method `Tsit5`
# is a fourth order Runge Kutta method.

#nb ?Tsit5

#nb #-

function S(A, u₀, t)
    problem = ODEProblem(ODEFunction(f), u₀, (0.0, t[end]), A)
    solve(problem, Tsit5(); saveat = t)
end

# Consider now the diffusion equation
#
# $$\frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial x^2}, \quad x \in
# \Omega = [0, 1]$$
#
# with diffusivity $\kappa > 0$, homogeneous Dirichlet boundary conditions $u(0, t) = u(1,
# t) = 0$, and initial conditions $u(x, 0) = u_0(x)$.

κ = 0.005
a = 0.0
b = 1.0
L = b - a

# The domain $\Omega$ may be discretized using a uniform grid $\mathbf{x} = (x_n)_{0 \leq n
# \leq N}$ of $N + 1$ equidistant points. We will also make a refined grid `xfine` for
# plotting.

xfine = LinRange(a, b, 1001)
N = 50
x = LinRange(a, b, N + 1)
Δx = L / N

# On the above grid, the diffusion operator $\frac{\partial^2}{\partial x^2}$ with constant
# Dirichlet boundary conditions may be approximated using the matrix
#
# $$\mathbf{D} = \frac{1}{\Delta x^2} \begin{pmatrix}
#     0 &  \dots &  \dots &  \dots & 0 \\
#     1 &     -2 &      1 &        &   \\
#       & \ddots & \ddots & \ddots &   \\
#       &        &      1 &     -2 & 1 \\
#     0 &  \dots &  \dots &  \dots & 0 \\
# \end{pmatrix}.$$
#
# This approximation is second order accurate:
#
# $$(\mathbf{D} \mathbf{u})_n = \frac{\partial^2 u}{\partial x^2}(x_n) + \mathcal{O}(\Delta
# x^2).$$

D = 1 / Δx^2 * spdiagm(-1 => fill(1.0, N), 0 => fill(-2.0, N + 1), 1 => fill(1.0, N))
D[1, :] .= 0 # Do not change first value
D[end, :] .= 0 # Do not change last value
plotmat(D)

# The semi-discrete solution $t \mapsto \mathbf{u}(t) = (u(x_n, t))_n \in \mathbb{R}^{N +
# 1}$ may be approximated using the solver $\mathbf{S}$:
#
# $$\mathbf{u}(t) \approx \mathbf{S}(\kappa \mathbf{D}, u_0(\mathbf{x}), t),$$
#
# i.e. we set $\mathbf{A} = \kappa \mathbf{D}$ in the ODE.

A_ref = κ * D
plotmat(A_ref)

# The diffusion operator $\frac{\partial^2}{\partial x^2}$ with homogeneous boundary
# conditions on $\Omega$ admit an eigenfunction basis $(X_k)_{k \in \mathbb{N}^*}$ with
#
# $$X_k(x) = \sqrt{\frac{2}{L}} \sin \left( \frac{\pi k x}{L} \right).$$
#
# The associated eigenvalues are given by $\lambda_k = -\frac{\pi^2 k^2}{L^2}$. Since these
# functions form a basis of $L^2(\Omega)$, all solutions to the diffusion equation with
# homogeneous boundary conditions may be written on the form
#
# $$u(x, t) = \sum_{k \in \mathbb{N}^*} c_k \exp \left( - \kappa \frac{\pi^2 k^2}{L^2}
# t \right) X_k(x),$$
#
# where the coefficients $(c_k)_{k \in \mathbb{N}^*}$ are determined by projecting the
# initial conditions onto the (orthogonal) eigenfunction basis:
#
# $$c_k = \int_\Omega u_0(x) X_k(x) \, \mathrm{d} x.$$
#
# In particular, we may use this formulation to generate exact solutions to the diffusion
# equation, by providing arbitrary coefficients. These solutions may used as training data
# for identifying the "unknown" discrete diffusion operator $\mathbf{A}$ (of which a very
# promising second-order accurate candidate is given by $\mathbf{A}_\text{ref} = \kappa
# \mathbf{D}$).

X(k, x) = √(2 / L) * sin(π * k * (x - a) / L)
p = plot(; xlabel = "x")
for k = 1:5
    plot!(p, xfine, X.(k, xfine); label = "k = $k")
end
p

#-

function create_solution(c, k)
    u(x, t) = sum(c * exp(-κ * (π * k / L)^2 * t) * X(k, x) for (c, k) in zip(c, k))
    ∂u∂t(x, t) =
        -sum(
            c * κ * (π * k / L)^2 * exp(-κ * (π * k / L)^2 * t) * X(k, x) for
            (c, k) ∈ zip(c, k)
        )
    u, ∂u∂t
end

# A particular solution is given below, containing three different frequencies. We may use
# this solution to test our solver, from now on referred to as the *full order model* (FOM).

k = [3, 7, 10]
c = [0.7, 0.3, 0.4]
u, ∂u∂t = create_solution(c, k)
p = plot();
tplot = LinRange(0.0, 1.0, 5)
sol = S(A_ref, u.(x, 0.0), tplot)
for (i, t) ∈ enumerate(tplot)
    plot!(p, xfine, u.(xfine, t); label = "t = $t", color = i) # Exact
    scatter!(p, x, sol[i]; label = nothing, markeralpha = 0.5, color = i) # FOM
end
p



# ## Learning the operator intrusively
#
# Before inferring the unknown operator, we need som "training" data to compare
# with. This will consist of snapshots of different initial conditions diffused
# for different durations. We will sample normally distributed random
# coefficients decaying with the frequency $k$, and put the results in a
# snapshot tensor of size $N \times n_\text{sample} \times n_t$.

tsnap = LinRange(0.0, 1.0, 51)[2:end]
nsample = 200
K = 50
k = 1:K
c = [randn(K) ./ k for _ = 1:nsample]
solutions = [create_solution(c, k) for c ∈ c]
u = [s[1] for s ∈ solutions]
∂u∂t = [s[2] for s ∈ solutions]
init = [u(x, 0.0) for x ∈ x, u ∈ u]
train = [u(x, t) for x ∈ x, u ∈ u, t ∈ tsnap]
∂train∂t = [∂u∂t(x, t) for x ∈ x, ∂u∂t ∈ ∂u∂t, t ∈ tsnap]

# We also need an instantaneous performance metric (loss/cost/objective
# function). This function should compare our predictions with a snapshots of
# the exact solutions. Here we will use a simple $L^2$-distance (mean squared
# error). Note that the `ODESolution` object behaves like an array of size $N
# \times n_\text{sample} \times n_t$, meaning that we solve for all the
# different initial conditions at the same time.

loss(A, u₀, uₜ, t) = sum(abs2, S(A, u₀, t) - uₜ) / prod(size(uₜ))
loss(A) = loss(A, init, train, tsnap)

# As an initial guess for the "unknown" operator $\mathbf{A}$ we will simply
# use an empty matrix.

A = zeros(N + 1, N + 1)

# We may also visualize the performance of our operator.

function ploterr(A, u, tplot = LinRange(0.0, 1.0, 5))
    sol = S(A, u.(x, 0.0), tplot)
    p = plot(; xlabel = "x")
    for (i, t) ∈ enumerate(tplot)
        plot!(p, xfine, u.(xfine, t); color = i, label = nothing)
        scatter!(p, x, sol[i]; label = "t = $t", color = i, markeralpha = 0.5)
    end
    p
end
ploterr(A_ref, u[6])

# A callback function is called after every iteration of gradient descent, allowing us to
# check the performance of our operator in real time during training. The return value
# `false` simply stops the function from stopping the iterations. We can already check how
# our initial guess for $\mathbf{A}$ performs.
#
# You may choose any of the callbacks below. For Jupyter notebooks, the two
# first options will print a plot after every iteration, which will quickly become very
# verbose. You may want to add a counter, plotting only every 10th itereation or so, or use
# the less verbose third callback option.

function callback(A, loss)
    println(loss)
    flush(stdout)
    display(
        plot(
            plotmat(A; title = "Predict"),
            plotmat(A_ref; title = "Reference");
            layout = (1, 2),
        ),
    )
    false
end
callback(A, loss(A))

#-

function callback(A, loss)
    println(loss)
    flush(stdout)
    display(ploterr(A, u[1], LinRange(0.0, 2.0, 5)))
    false
end
callback(A, loss(A))

#-

callback(A, loss) = (println(loss);
flush(stdout);
false)
callback(A, loss(A))

# The intrusive training consists of improving the operator through gradient
# descent applied to the loss function. The optimizer
# [`ADAM`](https://arxiv.org/abs/1412.6980) performs a first order gradient
# descent, but with some sophisiticated momentum terms exploiting the
# stochasticity of the loss function. For larger problems we could could use a
# subset of the different solutions $u$, time steps $t$ and spatial points $x$
# at every evaluation of `loss`, but for now we will just use the entire
# dataset.

result = DiffEqFlux.sciml_train(loss, A, ADAM(0.01); cb = callback, maxiters = 1000)
Afit = result.u
plotmat(Afit)

#-

ploterr(Afit, u[4])

# Notice that at no point did we explicitly specify the gradient of `loss`, `S`
# or even `f` with respect to the matrix `A`. Yet still we performed a gradient
# descent. Since the entire computational graph is composed of pure Julia code,
# automatic differentiation engines, in this particular case
# [Zygote](https://github.com/FluxML/Zygote.jl), can use the chain rule to
# compute gradients. We may access this gradient explicitly. Let us check for
# the initial guess first:

∂L∂A = first(Zygote.gradient(loss, A))
plotmat(∂L∂A)

# and for the final result:

∂L∂Afit = first(Zygote.gradient(loss, Afit))
plotmat(∂L∂Afit)

# The gradient of the fitted operator is indeed much closer to zero (indicating
# a possible minimum):

norm(∂L∂A)

#-

norm(∂L∂Afit)

# Note also that the optimizer used here is [ADAM](https://arxiv.org/abs/1412.6980), which
# is typically used to train neural networks. In fact, our ODE solver is not so different
# from a neural network. Here we used a 4/5 Runge Kutta solver, but consider for
# illustrative purposes a simple forward Euler scheme $\frac{\mathbf{u}^{n+1} -
# \mathbf{u}^n}{\Delta t} = \mathbf{A} \mathbf{u}^n$. It satisfies the definition of a
# "vanilla" neural network:
#
# | Neural network | ODE |
# | :------------: | :-: |
# | $$\mathbf{x} \to \left[ \operatorname{NN}_\theta \right] \to \mathbf{y}$$ | $$\mathbf{u}_0
# \to \left[ \frac{\mathrm{d} \mathbf{u}}{\mathrm{d} t} = \mathbf{A} \mathbf{u} \right] \to
# \mathbf{u}(T)$$ |
# | $$\mathbf{h}_0 = \mathbf{x}$$ | $$\mathbf{u}_0 = \mathbf{u}_0$$ |
# | $$\mathbf{h}_{k + 1} = \sigma_k(\mathbf{W}_k \mathbf{h}_k +
# \mathbf{b}_k)$$ | $$\mathbf{u}_{k + 1} = (\mathbf{I} + \Delta t_k \mathbf{A})
# \mathbf{u}_k$$ |
# | $$\mathbf{y} = \mathbf{h}_K$$ | $$\mathbf{u}(T) = \mathbf{u}_K$$ |
# | $$\underset{\theta}{\min} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathcal P} \|
# \mathbf{y}_\theta(\mathbf{x}) - \mathbf{y} \|^2$$ | $$\underset{\mathbf{A}}{\min}
# \mathbb{E}_{\mathbf{u} \sim \mathcal{U}} \| \mathbf{u}_\mathbf{A}(T) - \mathbf{u}(T)
# \|^2$$ |
#
#
#
# ### Adding a regularization
#
# We may also try out different loss functions. Since the resulting matrix from the above
# fit is quite dense, we could maybe enforce sparsity by adding penalization jump for going
# from zero to non-zero coefficients.

ℓ₁loss(A) = loss(A) + 1e-2 * sum(abs, A) / prod(size(A))
result = DiffEqFlux.sciml_train(ℓ₁loss, A, ADAM(0.01); cb = callback, maxiters = 1000)
A_ℓ₁ = result.u
plotmat(A_ℓ₁)

#-

ℓ₂loss(A) = loss(A) + 1e-2 * sum(abs2, A) / prod(size(A))
result = DiffEqFlux.sciml_train(ℓ₂loss, A, ADAM(0.01); cb = callback, maxiters = 1000)
A_ℓ₂ = result.u
plotmat(A_ℓ₂)

#-

ploterr(A_ℓ₁, u[1])

#-

ploterr(A_ℓ₂, u[1])

# Let us compare the resulting matrices:

plot(
    plotmat(Afit; title = "No reg"),
    plotmat(A_ℓ₁; title = "L¹"),
    plotmat(A_ℓ₂; title = "L²");
    layout = (1, 3),
    size = (900, 300),
)

# Indeed, the $L^1$-regularized matrix is sparser.

# ## Non-intrusive operator inference
#
# It is also possible to infer the operator $\mathbf{A}$ without ever computing the gradient
# of $\mathbf{S}$, using snapshot matrices only. Consider the above defined snapshot matrix
# $\mathbf{U}$ as well as its left hand side equivalent $\dot{\mathbf{U}}$ (containing
# snapshots of the time derivatives $\frac{\mathrm{d} \mathbf{u}}{\mathrm{d} t}(t_k)$). The
# operator that best satisfies the original ODE at the snapshots of interest should be the
# solution to the following minimization problem:
#
# $$\underset{\mathbf{A} \in \mathbb{R}^{N \times N}}{\min} \ell(\mathbf{A}),$$
#
# where $\ell$ is some performance metric, typically consisting of a data fitting term
# $\ell_\text{data}(\mathbf{A}) = \| \mathbf{A} \mathbf{U} - \dot{\mathbf{U}} \|_F^2$, where
# $\| \mathbf{X} \|_F = \sqrt{\sum_{i j} X_{i j}^2}$ is the Frobenius norm (we could use any
# discrete norm here). We would also like to add a regularization term $\ell_\text{reg}$ to
# enforce some expected behavior on the operator. Since the first and last component of the
# solution vector $\mathbf{u}$ are zero (because of the boundary conditions), the first and
# last columns of $\mathbf{A}$ do not affect the value of $\ell_\text{data}$. An
# $L^2$-regularization $\ell_\text{reg}(\mathbf{A}) = \lambda \| \mathbf{A} \|_F^2$ simply
# incites these two columns to take the value zero.
#
# The solution in the case of an $L^2$-regularization is given by
#
# $$\mathbf{A} = \underset{\mathbf{A} \in \mathbb{R}^{N \times N}}{\operatorname{argmin}} \|
# \mathbf{A} \mathbf{U} - \dot{\mathbf{U}} \|_F^2 + \lambda \| \mathbf{A} \|_F^2 =
# \dot{\mathbf{U}} \mathbf{U}^\mathsf{T} (\mathbf{U} \mathbf{U}^\mathsf{T} + \lambda
# \mathbf{I})^{-1}.$$

# Information from one single solution may not fully describe the dynamics of other
# solutions, as they may contain different frequencies. We may build an augmented snapshot
# matrix by concatenating snapshot matrices for many different initial conditions. Here we
# will use the exact solutions, but we could also have used the approximations as above.


U = reshape(train, N + 1, :)
∂U∂t = reshape(∂train∂t, N + 1, :)

plotmat(reshape(permutedims(train, (1, 3, 2)), N + 1, :); aspect_ratio = :none)

# We may zoom in on the first five solutions;

plotmat(
    reshape(permutedims(train, (1, 3, 2)), N + 1, :)[:, 1:(5 * length(tsnap))];
    aspect_ratio = :none,
)

#-

A_ls = ∂U∂t * U' / (U * U' + 1e-8I)
plotmat(A_ls)

#-

ploterr(A_ls, u[7], LinRange(0.0, 10.0, 5))



# # Proper orthogonal decomposition (POD)
#
# Above we learned the discrete diffusion operator in the canonical basis of
# $\mathbb{R}^N$. Another useful basis is obtained from a *proper orthogonal
# decomposition* (POD). It is determined from snapshot data of the solution at
# different time steps (`tsnap`) and possibly different initial conditions.
# Truncating this basis at a level $P \ll N$ will yield *the* basis of size $P$
# with the smallest error energy for the training data (among all possibile
# bases spanning $P$-dimensional subspaces of $L^2(\Omega)$).



# The POD basis is simply just a collection of left singular vectors of our snapshot matrix
# $\mathbf{U}$. We will keep the $P$ first basis functions (they are the most important, as
# `svd` orders them by decreasing singular value). The basis functions will be stored as
# columns in the matrix $\mathbf{\Phi} \in \mathbb{R}^{N \times P}$.

P = 20
decomp = svd(U)
Φ = decomp.U[:, 1:P]

# We may plot some POD modes.

p = plot(; xlabel = "x");
i = 0
for k ∈ [1, 3, 7]
    i += 1
    plot!(p, x, Φ[:, k]; label = "Mode $k", color = i)
    ## plot!(p, x, X.(k, x); label = "k = $k", color = i)
    ## plot!(p, x, XK[:, k]; label = nothing, linestyle = :dash, color = i)
end
p

# We may check the orthogonality by computing the inner product between each basis function
# pair.

plotmat(Φ'Φ)

# The matrix $\mathbf{Φ} \mathbf{Φ}^\mathsf{T}$ can be considered to be a so called
# "autoencoder", with "encoder" $\mathbf{Φ}^\mathsf{T}$ and "decoder" $\mathbf{Φ}$. The
# autoencoder should be closer to identity when keeping more modes, i.e. we may be tempted
# to write something like $\mathbf{Φ} \mathbf{Φ}^\mathsf{T} \underset{P \to N}{\to}
# \mathbf{I}$ (by abuse of mathematical notation).

plotmat(Φ * Φ')

# Projecting the full order model onto the POD basis yields the reduced order model
#
# $$\frac{\mathrm{d} \tilde{\mathbf{u}}}{\mathrm{d} t} = \tilde{\mathbf{A}}
# \tilde{\mathbf{u}},$$
#
# where $\tilde{\mathbf{u}}$ are the coordinates of the ROM solution in the POD basis and
# $\tilde{\mathbf{A}} = \mathbf{\Phi}^\mathsf{T} \mathbf{A} \mathbf{\Phi} \in \mathbb{R}^{P
# \times P}$ is the reduced order operator. Later, we will try to infer this operator
# directly from data. Note that the ROM solution is given by $\mathbf{u}_\text{ROM} =
# \mathbf{\Phi} \tilde{\mathbf{u}}$.
#
# Note also that the FOM and ROM have the same form, just different sizes (and "tilde"s
# appearing everywhere). The ROM solution may thus simply be computed by
#
# $$\mathbf{u}_\text{ROM}(t) = \mathbf{\Phi} \mathbf{S}(\tilde{\mathbf{A}},
# \mathbf{\Phi}^\mathsf{T} \mathbf{u}_0, t).$$

S_POD(A_pod, u₀, t) = Φ * S(A_pod, Φ'u₀, t)

# Let us compare the solution of the ROM and FOM:
tplot = LinRange(0.0, 1.0, 5)
sample = 3
sol = S(A_ref, u[sample].(x, 0.0), tplot)
sol_pod = S_POD(Φ' * A_ref * Φ, u[sample].(x, 0.0), tplot)
p = plot(; xlabel = "x");
for (i, t) ∈ enumerate(tplot)
    ## scatter!(p, x, u.(x, t); label = nothing, markeralpha = 0.5, color = i) # Exact
    plot!(p, x, sol[i]; label = "t = $t", color = i) # FOM
    scatter!(p, x, sol_pod[:, i]; label = nothing, markeralpha = 0.5, color = i) # ROM
end
p

# Try using fewer modes and see what happens!

# ## Learning the operator in the POD basis
#
# Similarly to the full order model case, we may fit the POD operator
# $\tilde{A}$ using intrusive and non-intrusive approaches.

# ### Non-intrusive approach

V = Φ'U
∂V∂t = Φ'∂U∂t

A_POD_ls = ∂V∂t * V' / (V * V' + 1e-8I)
plotmat(A_POD_ls)
plotmat(Φ * A_POD_ls * Φ')
ploterr(Φ * A_POD_ls * Φ', u[7], LinRange(0.0, 1.0, 5))

# ### Intrusive approach
#
# First we need to create snapshot tensors of the POD solutions/observations:

init_POD = Φ'init
train_POD = zeros(P, nsample, length(tsnap))
for i ∈ eachindex(tsnap)
    train_POD[:, :, i] = Φ' * train[:, :, i]
end

# We may also reuse the loss function and ODE solver from the full order case
# (only the sizes change).

loss_POD(A_POD) = loss(A_POD, init_POD, train_POD, tsnap)

result = DiffEqFlux.sciml_train(
    loss_POD,
    zeros(P, P),
    ADAM(0.01);
    cb = (A, l) -> callback(Φ * A * Φ', l),
    maxiters = 1000,
)

A_POD_fit = result.u

#-

plotmat(A_POD_fit)

#-

plotmat(Φ * A_POD_fit * Φ')

#-

ploterr(Φ * A_POD_fit * Φ', u[7], LinRange(0.0, 1.0, 5))


# ## Learning the operator in the eigenfunction basis
#
# In the continuous eigenfunction basis $(X_k)_k$, the continuous diffusion
# operator is an infinite diagonal matrix. Projecting the solution on to a
# truncated eigenfunction basis would thus result in a finite diagonal matrix.


# ### Non-intrusive approach

K_eig = 20
XK = √Δx .* X.((1:K_eig)', x)

#-

plotmat(XK'XK)

#-

plotmat(XK * XK')

#-

W = XK'U
∂W∂t = XK'∂U∂t

A_eig_ls = ∂W∂t * W' / (W * W' + 1e-8I)
plotmat(A_eig_ls)

#-

plotmat(XK * A_eig_ls * XK')

#-

ploterr(XK * A_eig_ls * XK', u[7], LinRange(0.0, 1.0, 5))


# ### Intrusive approach

init_eig = XK'init
train_eig = zeros(P, nsample, length(tsnap))
for i ∈ eachindex(tsnap)
    train_eig[:, :, i] = XK' * train[:, :, i]
end

#-

loss_eig(A) = loss(A, init_eig, train_eig, tsnap)

#-

result = DiffEqFlux.sciml_train(
    loss_eig,
    zeros(K_eig, K_eig),
    ADAM(0.01);
    cb = (A, l) -> callback(XK * A * XK', l),
    maxiters = 1000,
)
A_eig_fit = result.u

#-

plotmat(A_eig_fit)

#-

plotmat(XK * A_eig_fit * XK')

#-

ploterr(XK * A_eig_fit * XK', u[7], LinRange(0.0, 1.0, 5))


# ## Summary
#
# In this example, we considered a uniformly discretized diffusion equation
# with homogeneous Dirichlet boundary conditions for three discrete functional
# bases:
#
# * the canonical basis of $\mathbb{R}^N$: $\mathbf{I} = (\mathbf{e}_n)_n \in
#   \mathbb{R}^{N \times N}$
# * the eigenfunction basis of $\frac{\partial^2}{\partial x^2}$: $\mathbf{X} =
#   (X_k(\mathbf{x}))_k \in \mathbb{R}^{N \times K}$, $K \ll N$
# * the POD basis $\mathbf{\Phi} = (\mathbf{\phi}_p)_p \in \mathbb{R}^{N \times
#   P}$, $P \ll N$
#
# where the diffusion operator $\frac{\partial^2}{\partial x^2}$ was
# represented by $\mathbf{A}$, $\bar{\mathbf{A}}$, and $\tilde{\mathbf{A}}$
# respectively. These three operators, of respective sizes $N \times N$, $K
# \times K$, and $P \times P$, were then inferred using two methods:
#
# * intrusive inference, where the ODE-solver $\mathbf{S}$ needs to be available and
#   differentiable, and the operator is trained using gradient descent;
# * non-intrusive inference, using snapshot matrices only.
#
# For this simple test case, the latter option seems to work well. However, for
# nonlinear equations where the differential operator is nonlinear, a simple
# least squares fit will not do. Having access to differentiable ODE solver
# opens up a new world.
