import numpy as np
import matplotlib.pyplot as plt

def generate_concentric_hexagonal_lattice(d, n):
    """
    Generate a concentric hexagonal lattice with n layers around a central hexagon.

    Args:
        d (float): Side length of the hexagon.
        n (int): Number of layers surrounding the central hexagon.

    Returns:
        np.ndarray: Ordered array of unique nodal coordinates (x, y).
        list of np.ndarray: List of hexagons (each as an array of its centroid and vertices).
    """
    # Initialize lists for nodes and hexagons
    nodes = []
    hexagons = []

    # Define unit hexagon vertices (relative to the center)
    angle_offset = np.pi / 3  # 60 degrees in radians
    unit_hexagon = np.array([[d * np.cos(i * angle_offset), d * np.sin(i * angle_offset)] for i in range(6)])

    # Add the central hexagon
    centroid = [0, 0]
    hexagons.append(np.vstack((centroid, unit_hexagon)))
    nodes.extend(unit_hexagon.tolist())
    nodes.append(centroid)

    # Generate layers around the central hexagon
    for layer in range(1, n + 1):
        x = 0
        y = layer * np.sqrt(3) * d
        for side in range(6):  # Six sides of the hexagonal ring
            for step in range(layer):  # Number of hexagons on each side of the layer
                center = np.array([x, y])
                hexagon = np.vstack((center, unit_hexagon + center))
                hexagons.append(hexagon)
                nodes.extend((unit_hexagon + center).tolist())
                nodes.append(center.tolist())

                # Move to the next hexagon in the layer
                if side == 0:  # Down-right
                    x += 1.5 * d
                    y -= np.sqrt(3) / 2 * d
                elif side == 1:  # Down
                    x += 0
                    y -= np.sqrt(3) * d
                elif side == 2:  # Down-left
                    x -= 1.5 * d
                    y -= np.sqrt(3) / 2 * d
                elif side == 3:  # Up-left
                    x -= 1.5 * d
                    y += np.sqrt(3) / 2 * d
                elif side == 4:  # Up
                    x += 0
                    y += np.sqrt(3) * d
                elif side == 5:  # Up-right
                    x += 1.5 * d
                    y += np.sqrt(3) / 2 * d

    # Remove duplicate nodes and sort them by x, then y
    nodes = np.unique(np.round(nodes, decimals=8), axis=0)
    return nodes, hexagons


def generate_triangle_connectivity(nodes, hexagons):
    """
    Generate triangle connectivity for decomposing each hexagon into 6 triangles.

    Args:
        nodes (np.ndarray): Array of unique nodal coordinates (x, y).
        hexagons (list of np.ndarray): List of hexagons (each as an array of its centroid and vertices).

    Returns:
        np.ndarray: Connectivity array (n x 3), where each row contains node indices for a triangle.
    """
    connectivity = []

    # Build connectivity
    for hexagon in hexagons:
        centroid = hexagon[0]
        vertices = hexagon[1:]

        # Find the index of the centroid
        centroid_index = np.where(np.isclose(nodes, centroid, atol=1e-8).all(axis=1))[0][0]

        # Find the indices of the vertices
        vertex_indices = [np.where(np.isclose(nodes, vertex, atol=1e-8).all(axis=1))[0][0] for vertex in vertices]

        # Create 6 triangles (centroid + each edge of the hexagon)
        for i in range(len(vertex_indices)):
            triangle = [
                centroid_index,
                vertex_indices[i],
                vertex_indices[(i + 1) % len(vertex_indices)],
            ]
            connectivity.append(triangle)

    return np.array(connectivity)

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

def plot_hexagonal_lattice(nodes, hexagons):
    """
    Plot the hexagonal lattice with nodes and connectivity.

    Args:
        nodes (np.ndarray): Array of unique nodal coordinates (x, y).
        hexagons (list of np.ndarray): List of hexagons (each as an array of its centroid and vertices).
    """
    plt.figure(figsize=(8, 8))

    # Plot each hexagon
    for hexagon in hexagons:
        vertices = hexagon[1:]
        centroid = hexagon[0]

        # Draw hexagon edges
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'k-')

        # Draw centroid edges
        for vertex in vertices:
            plt.plot([centroid[0], vertex[0]], [centroid[1], vertex[1]], 'k-')

    # Plot nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', label='Vertices', s=20)
    centroids = np.array([hexagon[0] for hexagon in hexagons])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', label='Centroids', s=20)

    # Labeling and legend
    plt.title("Hexagonal Lattice with Nodes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
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
        self.dim = params[2]
        self.num_elem = params[3]
        self.num_nodes = params[4]
        self.num_elem_blk = params[5]

    # def gen_nodal_coords(self):
    #     """
    #     Generate the nodal x, y, z coordinates for the mesh
    #     """
    #     X, Y, Z = self.get_global_coords()
    #     coords = []
    #     for z in Z:
    #         for y in Y:
    #             for x in X:
    #                 coords.append((x, y, z))
    #     coords = np.array(coords)
    #     X = coords[:, 0]
    #     Y = coords[:, 1]
    #     Z = coords[:, 2]
    #     return X, Y, Z

    def set_elements(self):
        """
        Generate the elements in the mesh and the element blocks
        """
        # Materials and their corresponding element idx
    

    def gen_exofile(self):
        """
        Generate the .exo file
        """
        # Pre-processing steps to find the nodal coords and num elements.
        X, Y = self.get_coords()
        self.e = exo.exodusii_file('Hexagon.e', 'w')
        self.e.put_init('ExodusII File', self.dim, self.num_nodes, self.num_elem,
                        self.num_elem_blk, 0, 0)
        self.e.put_coord(X, Y)

        

    # def get_global_coords(self):
    #     """
    #     Get the x, y, z global coordinates
    #     """
    #     return self.x_coords, self.y_coords, self.z_coords
    
    def get_coords(self):
        """
        Get the x, y coordinates
        """ 
        return self.x_coords, self.y_coords
    
# Example usage
if __name__ == "__main__":
    d = 1  # Side length of the hexagon
    n = 2  # Number of layers around the central hexagon

    nodes, hexagons = generate_concentric_hexagonal_lattice(d, n)
    connectivity = generate_triangle_connectivity(nodes, hexagons)
    plot_hexagonal_lattice(nodes, hexagons)
    X, Y = nodes[:, 0], nodes[:, 1]
    # Generate the Z global coord array:
    main_divisions = np.array([0, 45, 65, 95, 125, 145, 155, 190])
    sub_divisions = np.array([9, 4, 6, 6, 4, 2, 7])
    Z = generate_z_intervals(main_divisions, sub_divisions)
    num_nodes = len(nodes)
    num_elem = len(connectivity)
    num_elem_blk = 12
    dim = 2
    print("Number of nodes:", num_nodes)
    print("Number of triangles:", num_elem)
    params = [X, Y, dim, num_elem, num_nodes, num_elem_blk]

