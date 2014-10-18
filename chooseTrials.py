from kassdirs import resFN, datFN

setname="080402-0-121"

dat = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))

#chosen = [251, 252, 254, 255, 257, 258, 259, 261, 266, 269, 270, 271, 272, 273, 274, 275, 276, 280, 282, 283, 291, 295, 296, 297, 300, 301, 306, 307, 308, 311, 316, 318, 319, 320, 322, 323, 324, 326, 329, 330, 331, 332, 333, 334, 336, 338, 340, 341, 342, 345, 353, 354, 355, 359, 360, 361, 362, 364, 368, 369, 372, 374, 375, 386, 390, 391, 395, 396, 397,]
#chosen = [1500, 1501, 1511, 1515, 1517, 1518, 1521, 1525, 1530, 1533, 1539, 1540, 1546, 1553, 1555, 1557, 1558, 1559, 1562, 1565, 1566, 1568, 1576, 1580, 1582, 1583, 1585, 1586, 1588, 1592, 1596, 1599, 1605, 1608, 1609, 1610, 1614, 1615, 1617, 1618, 1620, 1623, 1631, 1655, 1657, 1663, 1665, 1670, 1672, 1673, 1675, 1681, 1684, 1685, 1686, 1689, 1692, 1694, 1702, 1703, 1705, 1708, 1709, 1710, 1712, 1715, 1717, 1718, 1725, 1727, 1732, 1747, 1753, 1759, 1762, 1766, 1768, 1769, 1770, 1773, 1774, 1776, 1783, 1787, 1789, 1790, 1793]
chosen = [1001, 1007, 1008, 1009, 1023, 1028, 1029, 1030, 1033, 1034, 1036, 1039, 1040, 1041, 1044, 1045, 1046, 1048, 1049, 1053, 1055, 1056, 1058, 1060, 1066, 1067, 1081, 1082, 1083, 1085, 1086, 1088, 1100, 1101, 1121, 1128, 1129, 1134, 1139, 1141, 1155, 1156, 1160, 1161, 1162, 1180, 1184, 1185, 1187, 1189, 1192, 1193, 1194, 1196, 1201, 1206, 1209, 1210, 1225, 1255, 1256, 1257, 1266, 1267, 1276, 1279, 1280, 1282, 1283, 1284, 1286, 1289, 1290, 1296, 1298, 1302, 1305, 1311, 1312, 1313, 1315, 1317, 1327, 1343, 1347, 1365, 1369, 1370, 1372, 1375, 1379, 1383, 1390, 1395, 1396, 1399]

cdat = _N.empty((1000, 4*len(chosen)))

phs  = []
for ct in xrange(len(chosen)):
    cdat[:, 4*ct]   = dat[:, 4*chosen[ct]]
    cdat[:, 4*ct+1] = dat[:, 4*chosen[ct]+1]
    cdat[:, 4*ct+2] = dat[:, 4*chosen[ct]+2]
    cdat[:, 4*ct+3] = dat[:, 4*chosen[ct]+3]
    ts = _N.where(cdat[:, 2+4*ct] == 1)[0]
    phs.extend(cdat[ts, 3+4*ct])

csetname = "%s-c2" % setname
_N.savetxt(resFN("xprbsdN.dat", dir=csetname, create=True), cdat, fmt=("% .1f % .1f %d %.3f  " * len(chosen)))

_N.savetxt(resFN("chosen.dat", dir=csetname, create=True), _N.array(chosen), fmt="%d")
