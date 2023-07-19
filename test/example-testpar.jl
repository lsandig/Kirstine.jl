# A concrete parameter subtypes for tests where we do not need a full design problem

struct TestPar2 <: Parameter
    a::Float64
    b::Float64
end

struct TestPar3 <: Parameter
    a::Float64
    b::Float64
    c::Float64
end

Kirstine.dimension(p::TestPar2) = 2

Kirstine.dimension(p::TestPar3) = 3
