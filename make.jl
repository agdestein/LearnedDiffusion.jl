using Literate

problems = ["diffusion.jl"]
output = "notebooks"

for p ∈ problems
    # Literate.markdown(p, output)
    # Literate.script(p, output)
    Literate.notebook(p, output; execute = false)
end
