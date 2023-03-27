from swimfunction.behavior_annotation.RestPredictor import RestPredictor
import numpy

def test_find_rests():
    p = RestPredictor('smoothed_angles', 10)
    a = numpy.asarray([
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,5,6,7,8]
    ])
    expected = numpy.asarray([False, False, True, False, False, False, False, True, True])
    is_rest = p.find_rests_angles(a)
    assert numpy.all(expected == is_rest)

def test_find_rests_bulk():
    p = RestPredictor('smoothed_angles', 10)
    a = numpy.asarray([
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,55,6,7,8],
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,5,6,7,8],
        [1,2,3,4,5,6,7,8]
    ])
    expected = numpy.asarray([False, False, True, False, False, False, False, True, True])
    is_rest = p.find_rests_angles(a)
    assert numpy.all(expected == is_rest)

def test_find_rests_bulk_3d():
    p = RestPredictor('smoothed_angles', 10)
    a = numpy.asarray([
        [
            [1,2,3,4,5,6,7,8],
            [1,2,34,4,5,6,7,8],
            [1,2,3,4,56,6,7,8],
            [1,2,3,4,5,6,78,8]
        ], [
            [1,2,33,4,5,6,7,8],
            [1,2,3,4,5,6,7,8],
            [1,2,3,4,5,6,7,8],
            [1,2,3,4,5,6,7,8]
        ]
    ])
    expected = numpy.asarray([
        [False, False, False, False],
        [False, False, True, True]
    ])
    is_rest = p.find_rests_angles(a)
    assert numpy.all(expected == is_rest)

if __name__ == '__main__':
    test_find_rests()
    test_find_rests_bulk()
    test_find_rests_bulk_3d()
