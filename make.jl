using Literate

problems = ["diffusion.jl", "diffusion_complicated.jl"]
output = "notebooks"

for p ∈ problems
    # Literate.markdown(p, output)
    # Literate.script(p, output)
    Literate.notebook(p, output; execute = false)
end
