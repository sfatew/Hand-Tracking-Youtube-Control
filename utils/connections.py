POSE_CONNECTIONS = frozenset([(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)])

# Mapping from pose landmark IDs to file landmark IDs
poseid2fileid = {12: 0, 14: 1, 16: 2, 11: 3, 13: 4, 15: 5}

# Create file connections based on mapped pose IDs
POSE_CONNECTIONS_FILE_INDEX = [
    (poseid2fileid[a], poseid2fileid[b])
    for (a, b) in POSE_CONNECTIONS
]

#[(3, 0), (3, 4), (4, 5), (0, 1), (1, 2)]

"""
The 21 hand landmarks:
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20
"""

"Connection" 
HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])


handid2fileid = {a: a + 6 for a in range(21)}

#Mapping from joint_id to index_id of array

LEFT_HAND_CONNECTIONS_FILE_INDEX = [
    (handid2fileid[a], handid2fileid[b])
    for (a, b) in HAND_CONNECTIONS
]

RIGHT_HAND_CONNECTIONS_FILE_INDEX = [
    (handid2fileid[a] + 21, handid2fileid[b] + 21)
    for (a, b) in HAND_CONNECTIONS
]
