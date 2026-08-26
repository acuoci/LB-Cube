#!/bin/bash
#$ -cwd                 
#$ -N LBM_CUDA_Run           
#$ -j y                 
#$ -S /bin/bash         
#$ -l h_rt=02:00:00      
#$ -q hub.q             
#$ -pe mpi 1

# 1. Load necessary modules (Uncomment and adjust to CFDHub's specific modules)
# module load gcc/13.2.0
# module load cuda/12.2

echo "Starting C++23 LBM Simulation on CFDHub..."

# 2. Define CFDHub specific directories
# Replace 'username' and 'groupname' with your actual credentials
HOME_DIR="/home/username/lbm_solver"
SCRATCH_DIR="/global-scratch/bulk_pool/username/lbm_run"
ARCHIVE_DIR="/ARCHIVIO/groupname/lbm_results"

# 3. Prepare the high-speed scratch area for execution
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# 4. Copy the compiled executable from /home to the scratch disk
cp $HOME_DIR/build/lbm_executable .

# 5. Launch the solver (e.g., 128^3 grid, 5000 steps, using GPU backend)
./lbm_executable -nx 128 -ny 128 -nz 128 -steps 5000 -gpu

echo "End Parallel Run"

# 6. Post-processing: Move results to the long-term archive
mkdir -p $ARCHIVE_DIR
mv *.vtk $ARCHIVE_DIR/
mv *.csv $ARCHIVE_DIR/

echo "Results successfully transferred to /ARCHIVIO."