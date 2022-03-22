using SIAN, ModelingToolkit

# ODEModel format

@parameters k1 k2 eB
@variables t xA(t) xB(t) xC(t) eA(t) eC(t) y1(t) [output = true] y2(t) [output = true] y3(t) [output = true] y4(t) [output = true]
D = Differential(t)
eqs = [
    D(xA) ~ -k1 * xA,
    D(xB) ~ k1 * xA - k2 * xB,
    D(xC) ~ k2 * xB,
    D(eA) ~ 0,
    D(eC) ~ 0,
    y1 ~ xC,
    y2 ~ eA * xA + eB * xB + eC * xC,
    y3 ~ eA,
    y4 ~ eC
]

@named ode = ODESystem(eqs, t)
@time result = identifiability_ode(ode)
println(result)

###

# Goodwin

@parameters b c alpha beta g delta sigma
@variables t x1(t) x2(t) x3(t) x4(t) y(t) [output = true]
D = Differential(t)
eqs = [
    D(x1) ~ -b * x1 + 1 / (c + x4),
    D(x2) ~ alpha * x1 - beta * x2,
    D(x3) ~ g * x2 - delta * x3,
    D(x4) ~ sigma * x4 * (g * x2 - delta * x3) / x3,
    y ~ x1
]

@named goodwin = ODESystem(eqs, t)
@time res = identifiability_ode(goodwin; infolevel=0, p_mod=11863279, weighted_ordering=true)
@time res = identifiability_ode(goodwin; infolevel=0, p_mod=11863279, weighted_ordering=false)

###

# HIV model

ode = SIAN.@ODEmodel(
    x'(t) = lm - d * x(t) - beta * x(t) * v(t), # this one combined with v
    y'(t) = beta * x(t) * v(t) - a * y(t),
    v'(t) = k * y(t) - u * v(t),
    w'(t) = c * z(t) * y(t) * w(t) - c * q * y(t) * w(t) - b * w(t),
    z'(t) = c * q * y(t) * w(t) - h * z(t),
    y1(t) = w(t),
    y2(t) = z(t)
)

@time result = identifiability_ode(ode, get_parameters(ode); p_mod=11863279, p=0.99, infolevel=10)
@time result = identifiability_ode(ode, get_parameters(ode); p_mod=11863279, p=0.99, infolevel=10, weighted_ordering=true)
println(result)
###
