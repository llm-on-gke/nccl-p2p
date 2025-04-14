import os
import torch
import torch.distributed as dist
import argparse

def create_pairwise_groups_and_communicate():
    backend = 'nccl'
    dist.init_process_group(backend=backend)

    # Get rank, world size, and set device based on local rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    
    # Dictionary to store group handles this rank is part of
    my_groups_info = []
    my_groups_handles = {} # Store actual group handles keyed by sorted(i,j) tuple
    
    # Iterate through all possible pairs of ranks (i, j)
    print(f"Rank {rank}: Starting creation of {world_size * (world_size - 1)} potential pairwise groups...")
    for i in range(world_size):
        for j in range(world_size):
            if i == j:
                continue

            ranks_in_group = sorted([i, j])
            # Create group - this is collective
            group = dist.new_group(ranks=ranks_in_group)

            # Store the handle if this rank is part of the group
            if rank in ranks_in_group:
                group_pair = tuple(ranks_in_group)
                if group_pair not in my_groups_handles: # Store only once
                    my_groups_info.append(group_pair)
                    my_groups_handles[group_pair] = group

    print(f"Rank {rank}: Finished group creation loop. Acquired {len(my_groups_info)} group handles.")

    # Synchronize all processes before printing the summary
    print(f"Rank {rank}: Waiting on barrier after group creation...")
    dist.barrier()
    print(f"Rank {rank}: Passed barrier.")

    # Print a summary for the current rank
    print(f"\n--- Rank {rank} Summary ---")
    print(f"Rank {rank} is part of {len(my_groups_info)} pairwise groups:")
    my_groups_info.sort() # Sort for consistent output
    for group_pair in my_groups_info:
        print(f"  - Group {group_pair}")
    print(f"--- End Rank {rank} Summary ---\n")

    # --- Perform communication test within each group this rank belongs to ---
    print(f"Rank {rank}: Starting pairwise communication test across {len(my_groups_handles)} groups...")
    success_count = 0
    fail_count = 0

    # Ensure groups are processed in a consistent order for potentially easier debugging
    sorted_group_items = sorted(my_groups_handles.items())

    for group_pair, group_handle in sorted_group_items:
        rank1, rank2 = group_pair
        peer_rank = rank2 if rank == rank1 else rank1
        
        # Rank 0 initiates all sends and receives
        if rank == 0:
            # Send to all other ranks
            for dest_rank in range(1, world_size):
                if dest_rank == peer_rank:
                    tensor_to_send = torch.tensor([float(rank)], dtype=torch.float32, device=device)
                    print(f"Rank {rank}: Sending my rank ({rank}) to peer {dest_rank}...")
                    dist.send(tensor=tensor_to_send, dst=dest_rank, group=group_handle)
                    print(f"Rank {rank}: Send to {dest_rank} completed.")
                    success_count += 1
            
            # Receive from all other ranks
            for source_rank in range(1, world_size):
                if source_rank == peer_rank:
                    tensor_to_recv = torch.tensor([-1.0], dtype=torch.float32, device=device)
                    print(f"Rank {rank}: Receiving from peer {source_rank}...")
                    dist.recv(tensor=tensor_to_recv, src=source_rank, group=group_handle)
                    received_value = tensor_to_recv.item()
                    print(f"Rank {rank}: Received {received_value} from {source_rank}.")
                    success_count += 1
        else:
            # Other ranks send to and receive from rank 0
            if 0 in group_pair:  # Only communicate if this group includes rank 0
                # First receive from rank 0
                tensor_to_recv = torch.tensor([-1.0], dtype=torch.float32, device=device)
                print(f"Rank {rank}: Receiving from peer 0...")
                dist.recv(tensor=tensor_to_recv, src=0, group=group_handle)
                received_value = tensor_to_recv.item()
                print(f"Rank {rank}: Received {received_value} from rank 0.")
                success_count += 1
                
                # Then send to rank 0
                tensor_to_send = torch.tensor([float(rank)], dtype=torch.float32, device=device)
                print(f"Rank {rank}: Sending my rank ({rank}) to peer 0...")
                dist.send(tensor=tensor_to_send, dst=0, group=group_handle)
                print(f"Rank {rank}: Send to rank 0 completed.")
                success_count += 1

    print(f"Rank {rank}: Pairwise communication test finished. Successes: {success_count}, Failures: {fail_count}")

    # Barrier after communication loop
    print(f"Rank {rank}: Waiting on barrier after communication test...")
    dist.barrier()
    print(f"Rank {rank}: Passed barrier after communication test.")

    # Clean up the distributed environment
    print(f"Rank {rank}: Destroying process group.")
    dist.destroy_process_group()
    print(f"Rank {rank}: Process finished.")

if __name__ == "__main__":
    # Set RANK and WORLD_SIZE from Slurm variables
    #os.environ['RANK'] = os.environ['SLURM_PROCID']
    #os.environ['WORLD_SIZE'] = os.environ['SLURM_NPROCS']
    #os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']

    # Set MASTER_PORT
    if 'MASTER_PORT' not in os.environ:
      os.environ['MASTER_PORT'] = '29501' # Changed default port slightly just in case
      print(f"Rank {os.environ['RANK']}: MASTER_PORT not set, defaulting to {os.environ['MASTER_PORT']}")

    print(f"Rank {os.environ['RANK']}: Environment configured for PyTorch Distributed (NCCL).")
    print("Starting pairwise group creation and communication function...")
    create_pairwise_groups_and_communicate()
    print("Script finished.")