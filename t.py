exf("filter.py")

print base_q4atan(0.1, 0.1)
print _N.arctan2(0.1, 0.1) + _N.pi
print "---"
print base_q4atan(-0.1, 0.1)
print _N.arctan2(0.1, -0.1) + _N.pi
print "---"
print base_q4atan(-0.1, -0.1)
print _N.arctan2(-0.1, -0.1) + _N.pi
print "---"
print base_q4atan(0.1, -0.1)
print _N.arctan2(-0.1, 0.1) + _N.pi
