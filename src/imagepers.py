#!/usr/bin/python2

"""A simple implementation of persistent homology on 2D images."""

__author__ = "Stefan Huber <shuber@sthu.org>"


"""UnionFind.py

Union-find data structure. Based on Josiah Carlson's code,
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
with significant additional changes by D. Eppstein.
"""


class UnionFind:

    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def add(self, object, weight):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = weight

    def __contains__(self, object):
        return object in self.parents

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            assert(False)
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure.

        """
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.parents[r] = heaviest


def get(im, p):
    return im[p[0]][p[1]]


def iter_neighbors(p, w, h):
    y, x = p

    # 8-neighborship
    neigh = [(y+j, x+i) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    # 4-neighborship
    # neigh = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]

    for j, i in neigh:
        if j < 0 or j >= h:
            continue
        if i < 0 or i >= w:
            continue
        if j == y and i == x:
            continue
        yield j, i


def persistence(image):
    """
    Compute topological persistence of maxima in a 2D image using Union-Find.

    Parameters
    ----------
    image : ndarray
        2D image (e.g., chromatogram or heatmap) where pixel intensities
        indicate signal strength.

    Returns
    -------
    List[Tuple[Tuple[int, int], float, float, Tuple[int, int]]]
        A list of maxima with their persistence values.
        Each tuple contains:
            - coordinates of the dying maximum,
            - its birth intensity (when it was the highest),
            - persistence (birth - death intensity),
            - coordinates where the merge occurred.
        The list is sorted by decreasing persistence.

    Notes
    -----
    This function implements a 2D max-pooling persistence algorithm.
    High-persistence maxima correspond to dominant peaks, while 
    low-persistence ones can be considered noise.

    Example
    -------
    >>> output = persistence(my_image)
    >>> # Keep only the most persistent peaks
    >>> dominant_peaks = [entry for entry in output if entry[2] > threshold]
    """

    height, width = image.shape

    # Sort all pixel coordinates by intensity in descending order
    pixel_coords = [(i, j) for i in range(height) for j in range(width)]
    pixel_coords.sort(key=lambda p: get(image, p), reverse=True)

    # Maintains the growing sets
    uf = UnionFind()

    # Dictionary to store birth, persistence, and merge location of each 
    # maximum
    groups0 = {}

    def get_component_birth_intensity(coord):
        return get(image, uf[coord])

    # # Process pixels from highest to lowest intensity
    for i, coord in enumerate(pixel_coords):
        #print(p)
        current_intensity = get(image, coord)
        # Search the coordinates of the maximum for each pts in the
        # neighborhood
        neighborhood = [
            uf[n] for n in iter_neighbors(coord, width, height) if n in uf
            ]

        # Duplicates not allowed in set (merge the same points) in the
        # neighborhood et sort them by their comp birth (intensity)
        # Unique neighboring components, sorted by their birth intensity (descending)
        neighbor_components = sorted([(get_component_birth_intensity(c), c) for c in set(neighborhood)], reverse=True)
        
        # First pixel: initialize as a maximum
        if i == 0:
            groups0[coord] = (current_intensity, current_intensity, None)

        # Add current point as a new set in Union-Find
        uf.add(coord, -i) # -index ensures older entries have higher priority

        if len(neighbor_components) > 0:
            # Among the maxima (pts coordinates) which belong each pts in the neighborhood of the points get the coordinates of the highest
            # Ex: nc = [(44820405.83064588, (1154, 15)), (3495593.1969386637, (2096, 126))] -> (1154, 15)
            # Identify the dominant neighbor component
            dominant_coord = neighbor_components[0][1]
            # Merge the sets containing the items
            uf.union(dominant_coord, coord)

            #  Merge all other neighbor components with the dominant one
            for birth_intensity, neighbor_coord in neighbor_components[1:]:
                # If this neighbor component hasn't already been recorded 
                # (i.e., it's being merged into a stronger one), record its persistence
                if uf[neighbor_coord] not in groups0:
                    #print(i, ": Merge", uf[q], "with", oldp, "via", p)
                    #bl: intensity, v: point's intensity, p: point's coordinates
                    persistence_value = birth_intensity - current_intensity
                    groups0[uf[neighbor_coord]] = (birth_intensity, persistence_value , coord)
                uf.union(dominant_coord, neighbor_coord)

    groups0 = [(k, groups0[k][0], groups0[k][1], groups0[k][2]) for k in groups0]
    # Sort the maxima by their persistence (difference between their max intensity and their intensity)
    groups0.sort(key=lambda g: g[2], reverse=True)
    return groups0
