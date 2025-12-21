# SPDX-FileCopyrightText: 2023,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## concrete types for normal approximation ##

"""
    FisherMatrix

Normal approximation based on the maximum-likelihood approach.

The information matrix is obtained as the average of the Fisher information matrix with
respect to the design measure. Singular information matrices can occur.

See also the [mathematical background](math.md#Objective-Function).
"""
struct FisherMatrix <: NormalApproximation end

# In the simplest case, the normalized information is just the Fisher information matrix
# averaged wrt the design measure.
function informationmatrix!(afm::AbstractMatrix{<:Real}, na::FisherMatrix)
    return afm
end
