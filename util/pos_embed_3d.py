import numpy as np

def get_3d_sincos_pos_embed(embed_dim, grid_size, temporal_size, cls_token=False):
    """
    Generate 3D sine-cosine position embeddings for video inputs.

    Args:
        embed_dim (int): Dimension of the embedding.
        grid_size (int): Grid size for height and width dimensions. # (height or width) / patch_size
        temporal_size (int): Temporal size for the video sequence.
        cls_token (bool): Whether to include a class token embedding.

    Returns:
        pos_embed (np.ndarray): Position embedding of shape 
                                [temporal_size * grid_size * grid_size, embed_dim] 
                                or [1 + temporal_size * grid_size * grid_size, embed_dim] (if cls_token=True).
    """
    grid_t = np.arange(temporal_size, dtype=np.float32) # Temporal dimension [0,1,...,temporal_size-1]
    grid_h = np.arange(grid_size, dtype=np.float32) #  [0,1,...,grid_size-1]
    grid_w = np.arange(grid_size, dtype=np.float32) # same as grid_h
    
    # Create a 3D meshgrid for time, height, and width
    grid = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij') # 6 14 14 
    grid = np.stack(grid, axis=0)  # Shape: (3, temporal_size, grid_size, grid_size)

    # Reshape grid to [3, 1, temporal_size, grid_size, grid_size]
    grid = grid.reshape([3, temporal_size, grid_size, grid_size])
    
    # Generate the position embeddings using a sine-cosine function
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Compute the 3D sine-cosine position embeddings from a given grid.

    Args:
        embed_dim (int): Embedding dimension.
        grid (np.ndarray): Grid of shape [3, temporal_size, grid_size, grid_size].

    Returns:
        np.ndarray: The generated positional embeddings.
    """
    assert embed_dim % 6 == 0, "Embedding dimension must be divisible by 6 (for 3D: time, height, width)"

    emb_dim_per_axis = embed_dim // 6
    omega = 1.0 / (10000 ** (np.arange(emb_dim_per_axis, dtype=np.float32) / emb_dim_per_axis))

    # Expand omega to match the shape of grid dimensions
    omega = omega.reshape(1, 1, 1, -1)

    # Apply sine and cosine encoding to each dimension (temporal, height, width)
    pos_t = np.expand_dims(grid[0], axis=-1) * omega  # Expand last dimension
    pos_h = np.expand_dims(grid[1], axis=-1) * omega
    pos_w = np.expand_dims(grid[2], axis=-1) * omega

    pos_t = np.concatenate([np.sin(pos_t), np.cos(pos_t)], axis=-1)
    pos_h = np.concatenate([np.sin(pos_h), np.cos(pos_h)], axis=-1)
    pos_w = np.concatenate([np.sin(pos_w), np.cos(pos_w)], axis=-1)

    # Concatenate along embedding dimension and reshape
    pos_embed = np.concatenate([pos_t, pos_h, pos_w], axis=-1)
    pos_embed = pos_embed.reshape(-1, embed_dim)  # Flatten the embedding

    return pos_embed


# Example usage
embed_dim = 768
grid_size = 14  # 14x14 spatial grid
temporal_size = 6  # 6 frames in the video
pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size, temporal_size, cls_token=True)

print("Position Embedding Shape:", pos_embed.shape)
