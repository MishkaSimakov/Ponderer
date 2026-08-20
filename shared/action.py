# The policy commands armature voltage, not duty: the brick divides by the battery
# voltage it measures, so the same command means the same speed at any charge.
# The cap is what a battery still delivers under load, see logs/brick/measure-*.csv;
# above it the compensation would clip and voltage would leak back into the action.
VOLTS = 7.0
