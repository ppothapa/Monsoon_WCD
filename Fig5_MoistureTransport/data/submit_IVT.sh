#! /bin/ksh
#=============================================================================

# daint gpu batch job parameters
# ------------------------------
#SBATCH --job-name=IVT.run
#SBATCH --output=IVT.run.%j.o
#SBATCH --error=IVT.run.%j.o
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --uenv=netcdf-tools/2024:v1-rc1:/user-environment
#SBATCH --account=cwp03


#=============================================================================

source /capstor/store/cscs/exclaim/excp01/ppothapa/.env_icon/bin/activate

echo Submit Python Script 

python IVT_NH_highresolution.py 

echo END Python Script
