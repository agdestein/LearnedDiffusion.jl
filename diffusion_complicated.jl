# # Inferring diffusion operators - more complicated example
#
# In this example, we will consider the diffusion equation with Dirichlet and
# Neumann homogeneous boundary conditions, a variable diffusivity $\kappa(x)$,
# and a variable source term $s(x)$. You may change the profiles of $\kappa$
# and $s$.

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

# Since the problems will consist of inferring various matrices, we define a small helper
# function for visualizing our results.

plotmat(A; kwargs...) = heatmap(
    reverse(A; dims = 1);
    aspect_ratio = :equal,
    xlims = (1 / 2, size(A, 2) + 1 / 2),
    ylims = (1 / 2, size(A, 1) + 1 / 2),
    ## xticks = nothing,
    ## yticks = nothing,
    kwargs...,
)
plotmat(A::AbstractSparseMatrix; kwargs...) = plotmat(Matrix(A); kwargs...)


# ## Problem statement
#
# This time, we consider a linear ordinary differential equation (ODE)
# parameterized by an operator $\mathbf{A} \in \mathbb{R}^{(N + 1) \times (N +
# 1)}$ *and* a constant source term $\mathbf{s} \in \mathbb{R}^{N + 1}$:
#
# $$\frac{\mathrm{d} \mathbf{u}}{\mathrm{d} t} = \mathbf{f}(\mathbf{u},
# \mathbf{\theta}, t) := \mathbf{A} \mathbf{u} + \mathbf{s}, \quad
# \mathbf{u}(0) = \mathbf{u}_0$$
#
# where $\mathbf{u}_0 \in \mathbb{R}^{N + 1}$ are some initial conditions and
# $\mathbf{\theta} = \begin{pmatrix} \mathbf{A} & \mathbf{s} \end{pmatrix} \in
# \mathbb{R}^{(N + 1) \times (N + 2)}$. To solve this system, we will use the
# [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) package. It
# provides differentiable ODE solvers for problems defined by a parametrized
# ODE function $f$ defining the right hand side of the ODE for a given state
# $\mathbf{u}$, time $t$ and parameters $\mathbf{\theta}$.

function f(u, θ, t)
    A = @view θ[:, 1:end-1]
    s = @view θ[:, end]
    A * u .+ s
end
function f!(du, u, θ, t)
    A = @view θ[:, 1:end-1]
    s = @view θ[:, end]
    mul!(du, A, u)
    du .+= s
end

# Let us define the ODE solver (the in-place form `S!` may be faster for
# generating large datasets, but is not as easily differentiable):

function S(θ, u₀, t; kwargs...)
    problem = ODEProblem(ODEFunction(f), u₀, extrema(t), θ)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end
function S!(θ, u₀, t; kwargs...)
    problem = ODEProblem(ODEFunction(f!), u₀, extrema(t), θ)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

# Consider now the diffusion equation
#
# $$\frac{\partial u}{\partial t}(x, t) = \kappa(x) \frac{\partial^2 u}{\partial x^2}(x, t) + s(x), \quad x \in
# \Omega = [0, 1]$$
#
# with diffusivity $\kappa > 0$, source term $s$, and left homogeneous Dirichlet boundary conditions $u(0, t) = 0$, right homogeneous Neumann boundary conditions $\frac{\partial u}{\partial x}(1, t) = 0$, and initial conditions $u(x, 0) = u_0(x)$. *You may change the expressions for $\kappa$ and $s$*.

κ(x) = x ≤ 1 / 2 ? 0.01 : 0.001
source(x) = 1 / 5 * sin(2π * x) * (x ≤ 1 / 2)

# The domain $\Omega$ may be discretized using a uniform grid $\mathbf{x} = (x_n)_{0 \leq n
# \leq N}$ of $N + 1$ equidistant points.

N = 50
x = LinRange(0.0, 1.0, N + 1)
Δx = 1 / N

# On the above grid, the diffusion operator $\frac{\partial^2}{\partial x^2}$ with left
# Dirichlet and right Neumann boundary conditions may be approximated using the matrix
#
# $$\mathbf{D} = \frac{1}{\Delta x^2} \begin{pmatrix}
#     0 &  \dots &  \dots &  \dots & 0 \\
#     1 &     -2 &      1 &        &   \\
#       & \ddots & \ddots & \ddots &   \\
#       &        &      1 &     -2 & 1 \\
#       &        &      1 &     -2 & 1 \\
# \end{pmatrix}.$$
#
# A second order accurate diffusion stencil is given by

inds = [-1, 0, 1]
stencil = [1, -2, 1] / Δx^2

# Alternatively, a fourth order accurate diffusion stencil may be used. For higher order stencils, see
# https://en.wikipedia.org/wiki/Finite_difference_coefficient

inds = -2:2
stencil = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12] / Δx^2

#- We build diagonals from the stencil. Special treatment is needed for the boundaries.

A_ref =
    diagm(κ.(x)) * diagm((i => fill(s, N + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil))...)
A_ref[1, :] .= 0 # Left Dirichlet boundary conditions
A_ref[end, :] .= A_ref[end-1, :] # Right Neumann boundary conditions (first order)
## @. A_ref[end, :] = 2 / 3 * (2 * A_ref[end - 1, :] - 1 / 2 * A_ref[end - 2, :]) # Right Neumann boundary conditions (second order)
plotmat(A_ref)

# Discrete source term with boundary conditions

s_ref = source.(x)
s_ref[1] = 0
s_ref[end] = s_ref[end-1]
plot(x, s_ref)

# Reference parameters

θ_ref = [A_ref s_ref]

#-

function create_data(u₀, x, t, θ)
    A = θ[:, 1:end-1]
    s = θ[:, end]
    u = S!(θ, u₀, t; abstol = 1e-8, reltol = 1e-10)
    dudt = zeros(size(u))
    for i = 1:length(t)
        dudt[:, :, i] = A * u[i] .+ s
    end
    (; u, dudt)
end

#-

function apply_bc!(u)
    u[1] = 0
    u[end] = u[end-1]
    u
end

# Four particular solutions are given below. We may use this solution to test
# our solver, from now on referred to as the *full order model* (FOM).

u₁ = apply_bc!(@. sin(4π * x))
u₂ = apply_bc!(@. exp(-x) * x)
u₃ = apply_bc!(@. 1 / 3 ≤ x ≤ 2 / 3)
u₄ = apply_bc!(zeros(size(x)))
tplot = LinRange(0.0, 1.0, 5)
## u, dudt = create_data(reshape(u₁, :, 1), x, tplot, θ_ref)
## u, dudt = create_data(reshape(u₂, :, 1), x, tplot, θ_ref)
u, dudt = create_data(reshape(u₃, :, 1), x, tplot, θ_ref)
## u, dudt = create_data(reshape(u₄, :, 1), x, tplot, θ_ref)
pl = plot();
for (i, t) ∈ enumerate(tplot)
    plot!(pl, x, u[:, 1, i]; label = "t = $t")
end
pl



# ## Learning the operator intrusively
#
# Before inferring the unknown operator, we need som "training" data to compare
# with. This will consist of snapshots of different initial conditions diffused
# for different durations. We will sample normally distributed random
# coefficients decaying with the frequency $k$, and put the results in a
# snapshot tensor of size $(N + 1) \times n_\text{sample} \times n_t$.

tsnap = LinRange(0.0, 1.0, 51)[2:end]
nsample = 200
r = cumsum(randn(N + 1, nsample); dims = 1)
foreach(apply_bc!, eachcol(r))
train = create_data(r, x, tsnap, θ_ref)

# We also need an instantaneous performance metric (loss/cost/objective
# function). This function should compare our predictions with a snapshots of
# the exact solutions. Here we will use a simple $L^2$-distance (mean squared
# error). We also add regularizations for $\mathbf{A}$ and $\mathbf{s}$. Note
# that the `ODESolution` object behaves like an array of size $(N + 1) \times
# n_\text{sample} \times n_t$, meaning that we solve for all the different
# initial conditions at the same time.

function create_loss(u, t, λ = (1e-8, 1e-8))
    λᴬ, λˢ = λ
    nx, nu, _ = size(u)
    u₀ = u[:, :, 1]
    uₜ = u
    loss(θ) =
        sum(abs2, S(θ, u₀, t) - uₜ) / (nx * nu) +
        λᴬ / nx^2 * sum(abs2, θ[:, 1:end-1]) +
        λˢ / nx * sum(abs2, θ[:, end])
    loss
end

# As an initial guess for the "unknown" operators $\mathbf{\theta}$ we will
# simply use an empty matrix.

θ = zeros(N + 1, N + 2)

# We may also visualize the predictions of our operators.

function ploterr(θ, u, tplot = LinRange(0.0, 1.0, 5); θ_ref = θ_ref)
    sol_ref = S(θ_ref, u[:, :, 1], tplot)
    sol = S(θ, u[:, :, 1], tplot)
    pl = plot()
    for (i, t) ∈ enumerate(tplot)
        plot!(pl, x, sol_ref[i]; color = i, label = nothing)
        scatter!(pl, x, sol[i]; label = "t = $t", color = i, markeralpha = 0.5)
    end
    pl
end
ploterr(θ_ref, u₃)

# A callback function is called after every iteration of gradient descent, allowing us to
# check the performance of our operator in real time during training. The return value
# `false` simply stops the function from stopping the iterations. We can already check how
# our initial guess for $\mathbf{A}$ performs.

function callback(θ, loss)
    println(loss)
    flush(stdout)
    false
end

# The intrusive training consists of improving the operator through gradient
# descent applied to the loss function. The optimizer
# [`ADAM`](https://arxiv.org/abs/1412.6980) performs a first order gradient
# descent, but with some sophisiticated momentum terms exploiting the
# stochasticity of the loss function. For larger problems we could could use a
# subset of the different solutions $u$, time steps $t$ and spatial points $x$
# at every evaluation of `loss`, but for now we will just use the entire
# dataset.
# 
# https://diffeqflux.sciml.ai/dev/ControllingAdjoints/

loss = create_loss(train.u, tsnap, (1e-8, 1e-10))
θ_fit = θ
result = DiffEqFlux.sciml_train(loss, θ_fit, ADAM(0.01); cb = callback, maxiters = 1000)
θ_fit = result.u

#-

A = θ[:, 1:end-1]
A_fit = θ_fit[:, 1:end-1]
A_ref = θ_ref[:, 1:end-1]

#-

plot(
    plotmat(A; title = "Initial"),
    plotmat(A_fit; title = "Final"),
    plotmat(A_ref; title = "Reference");
    layout = (1, 3),
)

#-

plot(x, θ[:, end]; label = "Initial")
plot!(x, θ_fit[:, end]; label = "Final")
plot!(x, θ_ref[:, end]; label = "Reference")

#-

## ploterr(θ_fit, u₁)
## ploterr(θ_fit, u₂)
ploterr(θ_fit, u₃)
## ploterr(θ_fit, u₄)

# Notice that at no point did we explicitly specify the gradient of `loss`, `S`
# or even `f` with respect to the parameters `θ`. Yet still we performed a
# gradient descent. Since the entire computational graph is composed of pure
# Julia code, automatic differentiation engines, in this particular case
# [Zygote](https://github.com/FluxML/Zygote.jl), can use the chain rule to
# compute gradients. We may access this gradient explicitly. Let us compare the
# gradients:

dLdθ = first(Zygote.gradient(loss, θ))
dLdθ_fit = first(Zygote.gradient(loss, θ_fit))
dLdθ_ref = first(Zygote.gradient(loss, θ_ref))

dLdA = dLdθ[:, 1:end-1]
dLdA_fit = dLdθ_fit[:, 1:end-1]
dLdA_ref = dLdθ_ref[:, 1:end-1]

dLds = dLdθ[:, end]
dLds_fit = dLdθ_fit[:, end]
dLds_ref = dLdθ_ref[:, end]

plot(
    plotmat(dLdA; title = "Initial ($(norm(dLdA)))"),
    plotmat(dLdA_fit; title = "Final ($(norm(dLdA_fit)))"),
    plotmat(dLdA_ref; title = "Reference ($(norm(dLdA_ref)))");
    layout = (1, 3),
)



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

U = reshape(Array(train.u), N + 1, :)
P = 20
decomp = svd(U)
Φ = decomp.U[:, 1:P]
A_pod_ref = Φ' * A_ref * Φ
s_pod_ref = Φ's_ref
θ_pod_ref = [A_pod_ref s_pod_ref]
plotmat(A_pod_ref)

# We may plot some POD modes.

pl = plot();
for k ∈ [1, 3, 7]
    plot!(pl, x, Φ[:, k]; label = "Mode $k")
end
pl

# We may check the orthogonality by computing the inner product between each basis function
# pair.

plotmat(Φ'Φ)

# The matrix $\mathbf{Φ} \mathbf{Φ}^\mathsf{T}$ can be considered to be a so called
# "autoencoder", with "encoder" $\mathbf{Φ}^\mathsf{T}$ and "decoder" $\mathbf{Φ}$. The
# autoencoder should be closer to identity when keeping more modes, i.e. we may be tempted
# to write something like $\mathbf{Φ} \mathbf{Φ}^\mathsf{T} \underset{P \to N}{\to}
# \mathbf{I}$ (by abuse of mathematical notation).

plotmat(Φ * Φ')

# Projecting the full order model onto the POD basis yields the reduced order
# model
#
# $$\frac{\mathrm{d} \tilde{\mathbf{u}}}{\mathrm{d} t} = \tilde{\mathbf{A}}
# \tilde{\mathbf{u}} + \tilde{\mathbf{s}},$$
#
# where $\tilde{\mathbf{u}}$ are the coordinates of the ROM solution in the POD
# basis, $\tilde{\mathbf{A}} = \mathbf{\Phi}^\mathsf{T} \mathbf{A}
# \mathbf{\Phi} \in \mathbb{R}^{P \times P}$ is the reduced order operator, and
# $\tilde{\mathbf{s}} = \mathbf{\Phi}^\mathsf{T} \mathbf{s}. Later, we will try
# to infer this operator directly from data. Note that the ROM solution is
# given by $\mathbf{u}_\text{ROM} = \mathbf{\Phi} \tilde{\mathbf{u}}$.
#
# Note also that the FOM and ROM have the same form, just different sizes (and
# "tilde"s appearing everywhere). The ROM solution may thus simply be computed
# by
#
# $$\mathbf{u}_\text{ROM}(t) = \mathbf{\Phi} \mathbf{S}(\tilde{\mathbf{\theta}},
# \mathbf{\Phi}^\mathsf{T} \mathbf{u}_0, t).$$
#
# Let us compare the solution of the ROM and FOM:

## u₀ = u₁
## u₀ = u₂
u₀ = u₃
## u₀ = u₄

tplot = LinRange(0.0, 1.0, 5)
sol = S(θ_ref, u₀, tplot)
sol_pod = S(θ_pod_ref, Φ' * u₀, tplot)
p = plot();
for (i, t) ∈ enumerate(tplot)
    plot!(p, x, sol[i]; label = "t = $t", color = i) # FOM
    scatter!(p, x, Φ * sol_pod[:, i]; label = nothing, markeralpha = 0.5, color = i) # ROM
end
p


# ## Learning the operator in the POD basis
#
# Similarly to the full order model case, we may fit the POD operator
# $\tilde{A}$ using intrusive and non-intrusive approaches.

# First we need to create snapshot tensors of the POD solutions/observations:

u_pod = zeros(P, nsample, length(tsnap))
for i ∈ eachindex(tsnap)
    u_pod[:, :, i] = Φ' * train.u[:, :, i]
end

# We may also reuse the loss function and ODE solver from the full order case
# (only the sizes change).

loss_pod = create_loss(u_pod, tsnap)
θ_pod = zeros(P, P+1)
A_pod = θ_pod[:, 1:end-1]
s_pod = θ_pod[:, end]
result = DiffEqFlux.sciml_train(
    loss_pod,
    θ_pod,
    ADAM(0.01);
    cb = callback,
    maxiters = 1000,
)
θ_pod_fit = result.u
A_pod_fit = θ_pod_fit[:, 1:end-1]
s_pod_fit = θ_pod_fit[:, end]

#-

plot(
    plotmat(A_pod; title = "Initial"),
    plotmat(A_pod_fit; title = "Final"),
    plotmat(A_pod_ref; title = "Reference");
    layout = (1, 3),
)

#-

plot(
    plotmat(Φ * A_pod * Φ'; title = "Initial"),
    plotmat(Φ * A_pod_fit * Φ'; title = "Final"),
    ## plotmat(Φ * A_pod_ref * Φ'; title = "Reference");
    plotmat(A_ref; title = "Reference");
    layout = (1, 3),
)

#-

pl = plot(x, s_ref; label = "Reference")
plot!(pl, x, Φ * s_pod_ref; label = "Best approximation")
plot!(pl, x, Φ * s_pod; label = "Initial")
plot!(pl, x, Φ * s_pod_fit; label = "Final")
pl
