#include "finite_volume.h"

int FiniteVolume<double>::compute_dudt(const FlowStates<T>& flow_state,
                               const GridBlock<T>& grid,
                               ConservedQuantities<T>& dudt)
{
    apply_prep_reconstruction_bc();
    reconstruct(flow_state); 
}
