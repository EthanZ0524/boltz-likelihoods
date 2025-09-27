import mdtraj as md
import torch
import numpy as np
import torch.nn.functional as F
import re

from boltz.data import const

def _get_resnum(atom_name: str) -> int:
    """Helper function to extract residue ID from a full atom name."""
    match = re.search(r'(\d+)', atom_name)
    return int(match.group(1))

def pdb_to_boltz_coords(
    pdb_file: str,
    yaml_seq: str,
    atom_mask: torch.tensor,
    device: str,
    apply_padding: bool = True,
    frame_index: int = 0
):
    """Processes an input PDB file into its corresponding internal Boltz
    coordinate representation, as well as the corresponding element 
    symbols and masses (used for OpenMM system construction).
    
    Also performs a handful of sanity checks for the conversion.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file to convert to Boltz coordinates.
    
    yaml_seq : str
        Sequence provided to Boltz through the YAML input file. Needs
        to match with the protein sequence. Eg. YYDPETGTWY

    atom_mask : torch.tensor
        Tensor generated through running Pairformer of the inputted YAML
        seq or previously generated and loaded in tensors.hdf5 in 
        Boltz1._load_head_init(). The atom coordinates are padded to
        match atom_mask's dimensions to be properly shaped for Boltz.

    device : str

    apply_padding : bool, optional
        Whether to pad the coordinates to match atom_mask dimensions.
        Default is True.

    frame_index : int, optional
        Index of the frame to use from the PDB file (0-based). If the
        PDB contains multiple frames, this specifies which one to use.
        Default is 0 (first frame).

    Returns
    -------
    pdb_coords : torch.tensor of shape (n_padded_atoms, 3)
        Properly ordered and padded coordinates of provided PDB in 
        angstroms.

    elements : list
        A list of element names of pdb_coords' atoms.

    masses : list
        A list of masses (in daltons) of pdb_coords' atoms.
    """
    ref_atoms = const.ref_atoms
    has_ace = False

    yaml_seq_3 = [
        const.prot_letter_to_token[res] for res in yaml_seq
    ] # eg. [TYR, ARG, LEU].

    '''Loading the provided PDB and sanity checking its sequence.

    First, the PDB should not contain any non-canonical or modified 
    amino acids (other than potentially ACE or NME, which will be 
    handled further downstream), as Boltz does not provide atom 
    orderings for them. 

    Secondly, the PDB should match the inputted YAML sequence.
    '''
    init_pdb = md.load(pdb_file)
    prot = init_pdb.topology.select("protein")
    pdb = init_pdb.atom_slice(prot)
    
    # Validate frame_index
    if frame_index >= pdb.n_frames:
        print(
            f"Frame index {frame_index} is out of range for PDB {pdb_file} "
            f"which has {pdb.n_frames} frames. Using frame 0 instead."
        )
        frame_index = 0

    pdb_seq = []

    for residue in pdb.topology.residues:
        pdb_seq.append(residue.code)

    noncanonical_indices = [
        i + 1 for i, res in enumerate(pdb_seq) 
        if res is None and i != 0 and i != (len(pdb_seq) - 1)
    ] 

    if len(noncanonical_indices) > 0:
        print(
            f"Noncanonical residues found at the following "
            f"indices: {noncanonical_indices} for PDB {pdb_file}."
        )
        return None
    
    # Removing ACE and NME, if present. # TODO: this is not robust.
    if pdb_seq[-1] is None:
        pdb_seq = pdb_seq[:-1]

    if pdb_seq[0] is None:
        pdb_seq = pdb_seq[1:]
        has_ace = True

    if "".join(pdb_seq) != yaml_seq:
        print(f"Mismatched sequences for PDB {pdb_file}")
        return None
    
    '''If no errors occurred, we can process PDB coordinates safely.
    Iterating over the YAML sequence's residues, we use Boltz's 
    canonical internal ordering to fetch PDB atom coordinates in the 
    proper order.'''
    coord_list = []
    elements = []
    masses = []

    atom_coords = {
        str(list(pdb.topology.atoms)[i]): (
            pdb.xyz[frame_index][i],
            atom.element.symbol,
            atom.element.mass
        )
        for i, atom in enumerate(pdb.topology.atoms)

    } # Keys: eg. 'ACE1-CA'. 
    # Values: eg. ([0.2, 0.3, 0.1], 'C', 12.0)
    # Note: residue indices might not start on 1.

    # For PDB proteins, the chain might not start on residue 1.
    first_atom = str(list(pdb.topology.atoms)[0]) # eg. LYS1-N
    first_res_idx = _get_resnum(first_atom)

    start = first_res_idx + 1 if has_ace else first_res_idx

    for i, res in enumerate(yaml_seq_3, start=start):
        boltz_atom_ordering = ref_atoms[res]
        for atom in boltz_atom_ordering:
            atom_fullname = f'{res}{i}-{atom}'
            try:
                coord_list.append(atom_coords[atom_fullname][0])
                elements.append(atom_coords[atom_fullname][1])
                masses.append(atom_coords[atom_fullname][2])
            except Exception as e:
                raise Exception(
                    f"A Boltz canonical atom {atom} is missing in PDB {pdb_file}, "
                    f"residue {res}{i}."
                ) from e

    # Finding the padding dimension for coord_tensor.
    padding_dim = atom_mask.shape[-1]
    coord_tensor = torch.from_numpy(np.stack(coord_list)).to(device)
    rows_to_pad = padding_dim - coord_tensor.shape[0]
    if rows_to_pad < 0: # TODO: in principle, this shouldn't be necessary anymore
        print(
            f'Provided PDB {pdb_file} has more atoms than the '
            f'input conditioning tensors. Skipping.'
        )
        return None
    if apply_padding:
        pdb_coords = F.pad(coord_tensor, pad=(0, 0, 0, rows_to_pad))
    else:
        pdb_coords = coord_tensor
    pdb_coords = pdb_coords - pdb_coords.mean(dim=0, keepdim=True) # Centering to origin.
    pdb_coords *= 10 # Converting to angstroms.
    
    return pdb_coords, elements, masses

