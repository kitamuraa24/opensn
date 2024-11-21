import numpy as np
import matplotlib.pyplot as plt
import exodusii as exo

def generate_concentric_hexagonal_lattice(d, n):
    """
    Generate a concentric hexagonal lattice with n layers around a central hexagon.

    Args:
        d (float): Side length of the hexagon.
        n (int): Number of layers surrounding the central hexagon.

    Returns:
        np.ndarray: Ordered array of nodal coordinates (x, y), including vertices and centroids.
        list of np.ndarray: List of hexagons (each as an array of its vertices).
        no_hexagons (int): Number of hexagons in a layer 
    """
    # Initialize lists for nodes and hexagons
    nodes = []
    hexagons = []

    # Define unit hexagon vertices (relative to the center)
    angle_offset = np.pi / 3  # 60 degrees in radians
    unit_hexagon = np.array([
        [d * np.cos(i * angle_offset), d * np.sin(i * angle_offset)] for i in range(6)
    ])

    # Add the central hexagon
    hexagons.append(unit_hexagon)
    nodes.extend(unit_hexagon.tolist())  # Add vertices of the central hexagon
    nodes.append([0, 0])  # Centroid of the central hexagon

    # Generate layers around the central hexagon
    for layer in range(1, n + 1):
        # Start with the "north" hexagon of the current layer
        x = 0
        y = layer * np.sqrt(3) * d
        # Compute centers of all hexagons in this layer
        for side in range(6):  # Six sides of the hexagonal ring
            for step in range(layer):  # Number of hexagons on each side of the layer
                # Add the current hexagon's center
                center = np.array([x, y])
                hexagon = unit_hexagon + center
                hexagons.append(hexagon)
                nodes.extend(hexagon.tolist())  # Add vertices of the hexagon
                nodes.append(center.tolist())  # Add the centroid of the hexagon

                # Move to the next hexagon in the layer
                if side == 0:  # Moving "down-right"
                    x += 1.5 * d
                    y -= np.sqrt(3) / 2 * d
                elif side == 1:  # Moving "down"
                    x += 0
                    y -= np.sqrt(3) * d
                elif side == 2:  # Moving "down-left"
                    x -= 1.5 * d
                    y -= np.sqrt(3) / 2 * d
                elif side == 3:  # Moving "up-left"
                    x -= 1.5 * d
                    y += np.sqrt(3) / 2 * d
                elif side == 4:  # Moving "up"
                    x += 0
                    y += np.sqrt(3) * d
                elif side == 5:  # Moving "up-right"
                    x += 1.5 * d
                    y += np.sqrt(3) / 2 * d

    # Remove duplicate nodes and sort them
    nodes = np.unique(np.round(nodes, decimals=8), axis=0)  # Deduplicate and sort
    no_hexagons = len(hexagons)
    return nodes, hexagons, no_hexagons

def generate_z_intervals(main_divs, sub_divs):
    """
    Generate the Z coordinates given an array of the main divisions and their corresponding
    sub divisions.

    Args:
        main_divs (np.ndarray): Array of the main divisions.
        sub_divs (np.ndarray): Array of the sub divisions. Note: len - 1 of main_divs

    Returns:
        np.ndarray: Array of the z-coordinates of the mesh
    """
    Z = []
    for i in range(len(main_divs) - 1):
        start, end = main_divs[i], main_divs[i+1]
        num_sub = sub_divs[i]
        points = np.linspace(start, end, num_sub + 1)
        Z.extend(points)
    return np.unique(Z)

def count_num_wedges(no_hexagons, Z):
    """
    Calculate the total number of wedges (elements) in the mesh

    Args:
        no_hexagons (int): Number of hexagonal prisms in a z-layer
        Z (np.ndarray): Array of z coordinates

    Returns:
        num_wedges (int): Total number of wedge elements in mesh
    """
    num_wedges_per_zlayer = no_hexagons * 6
    num_zlayers = len(Z) - 1
    num_wedges = num_zlayers * num_wedges_per_zlayer
    return num_wedges

def plot_concentric_hexagonal_lattice(d, n):
    """
    Plot a concentric hexagonal lattice with n layers around a central hexagon.

    Args:
        d (float): Side length of the hexagon.
        n (int): Number of layers surrounding the central hexagon.
    """
    # Generate nodes and hexagons
    nodes, hexagons, _ = generate_concentric_hexagonal_lattice(d, n)

    # Plot hexagons
    plt.figure(figsize=(10, 10))
    for hexagon in hexagons:
        # Close the hexagon by repeating the first vertex
        hexagon_closed = np.vstack([hexagon, hexagon[0]])
        plt.plot(hexagon_closed[:, 0], hexagon_closed[:, 1], 'b-')

    # Plot nodes (vertices and centroids)
    plt.scatter(nodes[:, 0], nodes[:, 1], color='red', s=10, label="Vertices and Centroids")

    # Formatting the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Hexagonal Lattice with {n} Layers", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

class generate_exodusii:
    """
    Object to handle .exo mesh generation

    Args:
        X,Y,Z (np.ndarrays): Arrays of the global X,Y,Z mesh coordinates
        dim (int): 1-3, determines dimensionality of mesh
        num_elem (int): Total no. of elements in mesh
        num_nodes (int): Total no. of nodes in mesh
        num_elem_blk (int): Total no. of materials/zones in mesh.
    """
    def __init__(self, params):
        """
        Initialize object with parameters
        """
        self.x_coords = params[0]
        self.y_coords = params[1]
        self.z_coords = params[2]
        self.dim = params[3]
        self.num_elem = params[4]
        self.num_nodes = params[5]
        self.num_elem_blk = params[6]

    def gen_nodal_coords(self):
        """
        Generate the nodal x, y, z coordinates for the mesh
        """
        X, Y, Z = self.get_global_coords()
        coords = []
        for z in Z:
            for y in Y:
                for x in X:
                    coords.append((x, y, z))
        coords = np.array(coords)
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
        return X, Y, Z

    def gen_exofile(self):
        """
        Generate the .exo file
        """
        # Pre-processing steps to find the nodal coords and num elements.
        X, Y, Z = self.gen_nodal_coords()
        self.e = exo.exodusii_file('Hexagon.e', 'w')
        self.e.put_init('ExodusII File', self.dim, self.num_nodes, self.num_elem,
                        self.num_elem_blk, 0, 0)
        self.e.put_coord(X, Y, Z)
        self.e.describe()

    def get_global_coords(self):
        """
        Get the x, y, z global coordinates
        """
        return self.x_coords, self.y_coords, self.z_coords

if __name__ == "__main__":
    # Example Usage
    d = 7.5  # Side length of hexagon in cm
    n = 7    # Number of layers around the central hexagon

    # plot_concentric_hexagonal_lattice(d, n)

    # Output nodes for further use
    nodes, _, no_hexagons = generate_concentric_hexagonal_lattice(d, n)
    # Split this into X and Y global coord arrays
    X, Y = nodes[:,0], nodes[:,1]
    # Generate the Z global coord array:
    main_divisions = np.array([0, 45, 65, 95, 125, 145, 155, 190])
    sub_divisions = np.array([9, 4, 6, 6, 4, 2, 7])
    Z = generate_z_intervals(main_divisions, sub_divisions)
    # Calculate total number of nodes in mesh
    num_nodes = len(Z) * len(X) * len(Y)
    # Calculate total number of elements in mesh
    num_elem = count_num_wedges(no_hexagons, Z)
    # Determine number of materials/zones in mesh
    num_elem_blocks = 12
    # Define dimensionality of the problem (3D)
    dim = 3
    # Initialize mesh generation
    params = [X, Y, Z, dim, num_elem, num_nodes, num_elem_blocks]
    exo_file = generate_exodusii(params)
    e = exo_file.gen_exofile()