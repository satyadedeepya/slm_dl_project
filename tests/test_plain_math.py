from data.plain_math import has_latex, plain_math


def test_display_and_inline_math_become_plain_text():
    text = "Sum: $$ z = \\sum_{i} w_i x_i + b $$ so $z \\geq 0$ and output $\\boxed{1}$."
    out = plain_math(text)
    assert "z = Σᵢ wᵢ xᵢ + b" in out
    assert "z ≥ 0" in out and "output 1." in out
    assert not has_latex(out)


def test_fractions_scripts_and_greek():
    assert plain_math("$\\frac{\\partial L}{\\partial w}$") == "∂L/∂w"
    assert plain_math("$\\sigma(z) = \\frac{1}{1+e^{-z}}$") == "σ(z) = 1/(1+e^(-z))"
    assert plain_math("$O(n^2)$ and $O(n \\log_2 n)$") == "O(n²) and O(n log₂ n)"
    assert plain_math("$w_{new} = w_{old} - \\eta \\nabla L$") == "w_new = w_old - η ∇L"


def test_prices_are_not_math():
    assert plain_math("It costs $5 and $10.") == "It costs $5 and $10."


def test_matrices_and_aligned_equations():
    src = r"$$ \mathbf{z} = \begin{bmatrix} 0.5 & 1.0 \\ -0.8 & 0.5 \end{bmatrix} \begin{bmatrix} 2 \\ -1 \end{bmatrix} $$"
    assert plain_math(src).strip() == "z = [0.5, 1.0; -0.8, 0.5] [2; -1]"
    out = plain_math(r"\[ \begin{aligned} a &= 1 \\ b &= 2 \end{aligned} \]")
    assert "a = 1" in out and "b = 2" in out and "begin" not in out
