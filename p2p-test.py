import os
import torch
import torch.distributed as dist
import argparse
import time

def create_pairwise_groups_and_communicate():
    backend = 'nccl'
    dist.init_process_group(backend=backend)

    # Get rank, world size, and set device based on local rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    
    print(f"Rank {rank}: initialized with world_size={world_size} on device {device}")
    
    # Create all possible pairwise groups
    groups = {}
    all_pairs = []
    
    print(f"Rank {rank}: Creating {world_size * (world_size - 1) // 2} pairwise groups...")
    
    # Create unique pairwise groups (undirected pairs)
    for i in range(world_size):
        for j in range(i+1, world_size):  # Only create pairs (i,j) where i < j to avoid duplicates
            pair = (i, j)
            all_pairs.append(pair)
            groups[pair] = dist.new_group(ranks=[i, j])
    
    print(f"Rank {rank}: Created {len(groups)} pairwise groups.")
    
    # Barrier to ensure all groups are created
    dist.barrier()
    print(f"Rank {rank}: Passed barrier after group creation.")
    
    # Only rank 0 will initiate the communications
    if rank == 0:
        print(f"Rank 0: Starting to test all P2P channels ({len(all_pairs)*2} communications)...")
        
        # For each pair (i,j), we'll establish communication: 
        # i->j and j->i, orchestrated by rank 0
        for pair in all_pairs:
            i, j = pair
            group = groups[pair]
            
            # Skip if this rank is not in the pair
            if rank not in pair:
                continue
                
            # Rank 0 sends to other rank in the pair
            if rank == i:
                peer_rank = j
                # Send from rank 0 to peer
                tensor_to_send = torch.tensor([0.0], dtype=torch.float32, device=device)
                print(f"Rank {rank}: Sending to peer {peer_rank}...")
                dist.send(tensor=tensor_to_send, dst=peer_rank, group=group)
                print(f"Rank {rank}: Send to {peer_rank} completed.")
                
                # Receive from peer
                tensor_to_recv = torch.tensor([-1.0], dtype=torch.float32, device=device)
                print(f"Rank {rank}: Receiving from peer {peer_rank}...")
                dist.recv(tensor=tensor_to_recv, src=peer_rank, group=group)
                print(f"Rank {rank}: Received {tensor_to_recv.item()} from peer {peer_rank}.")
    else:
        # For non-zero ranks, wait for communication initiated by rank 0
        for pair in all_pairs:
            i, j = pair
            
            # Skip if this rank is not in the pair
            if rank not in pair:
                continue
                
            # This rank must be j since it's not 0 and it's in the pair
            if rank == j and i == 0:  # Only handle pairs with rank 0
                group = groups[pair]
                
                # First receive from rank 0
                tensor_to_recv = torch.tensor([-1.0], dtype=torch.float32, device=device)
                print(f"Rank {rank}: Receiving from rank 0...")
                dist.recv(tensor=tensor_to_recv, src=0, group=group)
                print(f"Rank {rank}: Received {tensor_to_recv.item()} from rank 0.")
                
                # Then send back to rank 0
                tensor_to_send = torch.tensor([float(rank)], dtype=torch.float32, device=device)
                print(f"Rank {rank}: Sending to rank 0...")
                dist.send(tensor=tensor_to_send, dst=0, group=group)
                print(f"Rank {rank}: Send to rank 0 completed.")
    
    # Wait for all communications to complete
    print(f"Rank {rank}: Waiting on barrier after communication test...")
    dist.barrier()
    print(f"Rank {rank}: Passed barrier after communication test.")

    # Clean up
    print(f"Rank {rank}: Destroying process group.")
    dist.destroy_process_group()
    print(f"Rank {rank}: Process finished.")

if __name__ == "__main__":
    # Set MASTER_PORT
    if 'MASTER_PORT' not in os.environ:
      os.environ['MASTER_PORT'] = '29501' # Changed default port slightly just in case
      print(f"MASTER_PORT not set, defaulting to {os.environ['MASTER_PORT']}")

    print(f"Environment configured for PyTorch Distributed (NCCL).")
    print("Starting pairwise group creation and communication function...")
    create_pairwise_groups_and_communicate()
    print("Script finished.")