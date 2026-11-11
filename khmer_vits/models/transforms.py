"""Piecewise rational-quadratic spline (Durkan et al., 2019), used by the duration predictor."""
import numpy as np
import torch
import torch.nn.functional as F


MIN_BIN_WIDTH, MIN_BIN_HEIGHT, MIN_DERIVATIVE = 1e-3, 1e-3, 1e-3


def _searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = torch.cat([bin_locations[..., :-1], bin_locations[..., -1:] + eps], dim=-1)
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def _rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                               inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0):
    num_bins = unnormalized_widths.shape[-1]
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = MIN_BIN_WIDTH + (1 - MIN_BIN_WIDTH * num_bins) * widths
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths = torch.cat([torch.full_like(cumwidths[..., :1], left), cumwidths[..., 1:-1],
                           torch.full_like(cumwidths[..., :1], right)], dim=-1)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    derivatives = MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = MIN_BIN_HEIGHT + (1 - MIN_BIN_HEIGHT * num_bins) * heights
    cumheights = F.pad(torch.cumsum(heights, dim=-1), pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights = torch.cat([torch.full_like(cumheights[..., :1], bottom), cumheights[..., 1:-1],
                            torch.full_like(cumheights[..., :1], top)], dim=-1)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = _searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = ((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
             + input_heights * (input_delta - input_derivatives))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = -input_delta * (inputs - input_cumheights)
        discriminant = (b.pow(2) - 4 * a * c).clamp(min=0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        return outputs, -(torch.log(derivative_numerator) - 2 * torch.log(denominator))
    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
    denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                 * theta_one_minus_theta)
    outputs = input_cumheights + numerator / denominator
    derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                 + 2 * input_delta * theta_one_minus_theta
                                                 + input_derivatives * (1 - theta).pow(2))
    return outputs, torch.log(derivative_numerator) - 2 * torch.log(denominator)


def piecewise_rational_quadratic(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                                 inverse=False, tail_bound=5.0):
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outputs = inputs.clone()  # linear tails: identity outside the interval
    logabsdet = torch.zeros_like(inputs)
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = float(np.log(np.exp(1 - MIN_DERIVATIVE) - 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant
    out_in, lad_in = _rational_quadratic_spline(
        inputs[inside], unnormalized_widths[inside, :], unnormalized_heights[inside, :],
        unnormalized_derivatives[inside, :], inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound)
    outputs = outputs.masked_scatter(inside, out_in)
    logabsdet = logabsdet.masked_scatter(inside, lad_in)
    return outputs, logabsdet
