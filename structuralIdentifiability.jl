using StructuralIdentifiability, ModelingToolkit
using DifferentialEquations, Distributions, Plots

plotly()

###
@parameters A B C
@variables t x1(t) x2(t) y(t)
D = Differential(t)

eqs = [
    D(x1) ~ -A^2 * x1 + B
    D(x2) ~ -A * C * x2
]
@named simple_example = ODESystem(eqs, t)
mq = [y ~ x1]

# ====
@time result_local = assess_local_identifiability(simple_example, measured_quantities=mq)
println(result_local)

# ====
@time result_full = assess_identifiability(simple_example, measured_quantities=mq)
println(result_full)

initial_conditions = [
    x1 => 1 + rand(Uniform(1, 2)),
    x2 => 1 + rand(Uniform(1, 2)),
]
tspan = (0, 50)
params = Dict(
    A => rand(Uniform(0, 1)),
    B => rand(Uniform(0, 1)),
    C => rand(Uniform(0, 1))
)

problem = ODEProblem(simple_example, initial_conditions, tspan, params)
sol = solve(problem)
cc = params[C]

plot(sol, vars=(t, x1), lab="C=$cc")

params[C] += 0.2
problem = ODEProblem(simple_example, initial_conditions, tspan, params)
sol2 = solve(problem)

cc = params[C]
plot!(sol2, vars=(t, x1), lab="C=$cc")

###
# HIV model
# D. Wodarz and M. A. Nowak, Specific therapy regimes could lead to long-term immunological control of HIV
# Proceedings of the National Academy of Sciences, 96 (25), 1999
# 10.1073/pnas.96.25.14464

ode = @ODEmodel(
    x'(t) = lm - d * x(t) - beta * x(t) * v(t),
    y'(t) = beta * x(t) * v(t) - a * y(t),
    v'(t) = k * y(t) - u * v(t),
    w'(t) = c * z(t) * y(t) * w(t) - c * q * y(t) * w(t) - b * w(t),
    z'(t) = c * q * y(t) * w(t) - h * z(t),
    y1(t) = w(t),
    y2(t) = z(t)
)

@time result_all = assess_identifiability(ode)

###
@parameters k1 k2 eB
@variables t xA(t) xB(t) xC(t) eA(t) eC(t) y1(t) y2(t) y3(t) y4(t)
D = Differential(t)
eqs = [
    D(xA) ~ -k1 * xA,
    D(xB) ~ k1 * xA - k2 * xB,
    D(xC) ~ k2 * xB,
    D(eA) ~ 0,
    D(eC) ~ 0,
]

measured = [
    y1 ~ xC,
    y2 ~ eA * xA + eB * xB + eC * xC,
    y3 ~ eA,
    y4 ~ eC
]
@named ode = ODESystem(eqs, t)
result = assess_identifiability(ode, measured_quantities=measured, funcs_to_check=[k1 * k2, k1 + k2])

println(result)