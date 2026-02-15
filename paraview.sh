alias paraview='LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ":" "\n" | grep -v oneapi | paste -sd ":") paraview'
paraview