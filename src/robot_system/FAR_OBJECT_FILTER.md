# Far object filtering

The system ignores physical detections whose horizontal distance from
`base_link` is greater than 0.40 m.

The filter is enforced in three places:

1. workspace point-cloud object extraction;
2. object instance registration;
3. planning target resolution.

Configuration:

```env
SIMPLE_OBJECT_MAX_BASE_DISTANCE_M=0.40
PLANNER_MAX_TARGET_BASE_DISTANCE_M=0.40
INSTANCE_MATCH_MAX_DISTANCE_M=0.06
```

The scanner still uses the full configured table area for table-plane
estimation. Only object candidates outside the reach radius are discarded.
