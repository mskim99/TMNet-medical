from numpy import array, cross, pi, cos, arccos as acos
import numpy.linalg as LA

def scalar(v1, v2):
    """
    calculates the scalar product of two vectors
    v1 and v2 are numpy.array objects.
    returns a float for a one-dimensional array.
    """
    return sum(v1 * v2)


def angle(v1, v2):
    """
    calculates the angle between two vectors.
    v1 and v2 are numpy.array objects.
    returns a float containing the angle in radians.
    """
    length_product = LA.norm(v1) * LA.norm(v2)
    if length_product == 0:
        raise ValueError("Cannot calculate angle for vectors with length zero")
    cosine = scalar(v1, v2) / length_product
    angle = acos(cosine)
    return angle


def calc_angle(vec1, vec2, vec3):
    """Calculates a flat angle from three coordinates."""
    if len(vec1) == 3:
        v1, v2, v3 = map(create_vector, [vec1, vec2, vec3])
    else:
        v1, v2, v3 = map(create_vector2d, [vec1, vec2, vec3])
    v12 = v2 - v1
    v23 = v2 - v3
    return angle(v12, v23)


def create_vector2d(vec):
    """Returns a vector as a numpy array."""
    return array([vec[0], vec[1]])


def create_vector(vec):
    """Returns a vector as a numpy array."""
    return array([vec[0], vec[1], vec[2]])


def create_vectors(vec1, vec2, vec3, vec4):
    """Returns dihedral angle, takes four
    Scientific.Geometry.Vector objects
    (dihedral does not work for them because
    the Win and Linux libraries are not identical.
    """
    return map(create_vector, [vec1, vec2, vec3, vec4])


def dihedral(vec1, vec2, vec3, vec4):
    """
    Returns a float value for the dihedral angle between
    the four vectors. They define the bond for which the
    torsion is calculated (~) as:
    V1 - V2 ~ V3 - V4
    The vectors vec1 .. vec4 can be array objects, lists or tuples of length
    three containing floats.
    For Scientific.geometry.Vector objects the behavior is different
    on Windows and Linux. Therefore, the latter is not a featured input type
    even though it may work.

    If the dihedral angle cant be calculated (because vectors are collinear),
    the function raises a DihedralGeometryError
    """
    # create array instances.
    v1, v2, v3, v4 = create_vectors(vec1, vec2, vec3, vec4)
    all_vecs = [v1, v2, v3, v4]

    # rule out that two of the atoms are identical
    # except the first and last, which may be.
    for i in range(len(all_vecs) - 1):
        for j in range(i + 1, len(all_vecs)):
            if i > 0 or j < 3:  # exclude the (1,4) pair
                equals = all_vecs[i] == all_vecs[j]
                if equals.all():
                    raise ValueError("Vectors #%i and #%i may not be identical!" % (i, j))

    # calculate vectors representing bonds
    v12 = v2 - v1
    v23 = v3 - v2
    v34 = v4 - v3

    # calculate vectors perpendicular to the bonds
    normal1 = cross(v12, v23)
    normal2 = cross(v23, v34)

    # check for linearity
    if LA.norm(normal1) == 0 or LA.norm(normal2) == 0:
        raise ValueError("Vectors are in one line; cannot calculate normals!")

    # normalize them to length 1.0
    normal1 = normal1 / LA.norm(normal1)
    normal2 = normal2 / LA.norm(normal2)

    # calculate torsion and convert to degrees
    torsion = angle(normal1, normal2) * 180.0 / pi

    # take into account the determinant
    # (the determinant is a scalar value distinguishing
    # between clockwise and counter-clockwise torsion.
    if scalar(normal1, v34) >= 0:
        return torsion
    else:
        torsion = 360 - torsion
        if torsion == 360: torsion = 0.0
        return