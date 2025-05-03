using HopTB
using DelimitedFiles
using StaticArrays
using HDF5
using LinearAlgebra

@everywhere function extract_chemical_potential_from_scfout(scfout_path)
    # Open the SCFOUT file
    openmx_file = open(scfout_path, "r")
    
    # Read the file line by line
    for line in eachline(openmx_file)
        # Look for the line containing "Chemical potential" or a similar keyword
        if occursin(r"Chemical potential \(Hartree\)", line)
            # Extract the chemical potential value from the line
            # Assuming the chemical potential is the last number on the line
            chemical_potential = parse(Float64, split(line)[end])
            close(openmx_file)
            return chemical_potential
        end
    end
    
    # If no chemical potential line is found, return an error message
    close(openmx_file)
    throw("Chemical potential not found in SCFOUT file.")
end
Hartree2ev = 27.2114
pattern = r".*\.scfout$"  # Regex to match *.vasp.run
files = filter(f -> occursin(pattern, f), readdir("."))
println(files)
scfout_path = files[1]
tm = HopTB.Interface.createmodelopenmx(scfout_path)
μ = extract_chemical_potential_from_scfout(scfout_path) * Hartree2ev 
println(μ)
sm = SharedTBModel(tm)
kgrid_density = 0.05  # in units of 1/Å
# Compute mesh size
meshsize = [round(Int, norm(tm.rlat[i, :]) / kgrid_density) for i in 1:3]
meshsize = [x % 2 == 0 ? x : x + 1 for x in meshsize]  
println("Calculated mesh size: ", meshsize)
ωs = collect(range(0, stop=30, length=3001))
gauss_width = 0.05
# Initialize a 3D array to store results
num_ωs = length(ωs)
# Initialize the tensor to store real and imaginary parts separately
tensor = zeros(Float64, 3, 3, num_ωs, 2)  # 4D tensor: 3x3 grid, num_ωs frequencies, 2 for real/imag parts

# Loop over unique pairs (alpha, beta) with alpha ≤ beta
for alpha in 1:3
    for beta in alpha:3  # Only iterate where beta >= alpha
        println("Calculating for alpha = $alpha, beta = $beta")
        
        # Compute permittivity tensor element
        sc = HopTB.Optics.getpermittivity(sm, alpha, beta, ωs, μ, meshsize, ϵ=gauss_width)
        
        # Store the real and imaginary parts separately
        tensor[alpha, beta, :, 1] .= real(sc)
        tensor[alpha, beta, :, 2] .= imag(sc)
        
        # Copy to symmetric position if alpha ≠ beta
        if alpha != beta
            tensor[beta, alpha, :, 1] .= real(sc)
            tensor[beta, alpha, :, 2] .= imag(sc)
        end
    end
end

# Save the real and imaginary parts to a file
open("permittivity.dat", "w") do f
    for i in 1:num_ωs
        row = [ωs[i]]  # Start with the frequency (omega)

        # Alternate real and imaginary parts for each (alpha, beta)
        for alpha in 1:3
            for beta in 1:3
                push!(row, tensor[alpha, beta, i, 1])  # Real part
                push!(row, tensor[alpha, beta, i, 2])  # Imaginary part
            end
        end
        
        writedlm(f, [row], " ")
    end
end

println("done")

