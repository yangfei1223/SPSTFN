# -*- coding:utf-8 -*-
# trainval text
f = open('../data/trainval.txt', 'w')
for i in range(95):
    f.write('um_%06d\n' % (i))
for i in range(96):
    f.write('umm_%06d\n' % (i))
for i in range(98):
    f.write('uu_%06d\n' % (i))
f.close()

# train text
f = open('../data/train.txt', 'w')
for i in range(79):
    f.write('um_%06d\n' % i)
for i in range(80):
    f.write('umm_%06d\n' % i)
for i in range(82):
    f.write('uu_%06d\n' % i)
f.close()

# val text
f = open('../data/val.txt', 'w')
for i in range(79, 95):
    f.write('um_%06d\n' % i)
for i in range(80, 96):
    f.write('umm_%06d\n' % i)
for i in range(82, 98):
    f.write('uu_%06d\n' % i)

# test text
f = open('../data/test.txt', 'w')
for i in range(96):
    f.write('um_%06d\n' % i)
for i in range(94):
    f.write('umm_%06d\n' % i)
for i in range(100):
    f.write('uu_%06d\n' % i)
f.close()
