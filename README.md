# VirxEPH

## PacketHeuristic

A class where you just pass in the game tick packet and it handles the rest.

For an example implementation see `external.py`.

### __init__(self, threshold: float=0.8, gain: float=0.21, loss: float=0.005, unpause_delay: float=1.5, ignore_indexes: List[int]=[])

Initializes the packet heuristics. If you don't have anything specific in mind, just leave all of the values at their defaults.

+ `threshold` - The dividing point between `True` and `False`
+ `gain` - The amount gained from a completed action
+ `loss` - The amount lost every second that an action isn't completed
+ `unpause_delay` - How long to wait after kickoff before starting (some) heuristics
+ `ignore_indexes` - A list of the indexes of the cars that you want to heuristics to ignore

### add_tick(self, packet: GameTickPacket, ball_prediction_struct: BallPrediction) -> bool

Adds a new game tick to the heuristics. The more the better, but don't add the same game tick twice!

+ `packet` - The game tick packet.
+ `ball_prediction_struct` - The ball prediction struct. (e.x. pass in the struct returned from `self.get_ball_prediction_struct()`)

Returns `False` if the packet wasn't added to the heuristics due to the game being paused, and `True` if it was.

### get_car(self, car_name: str) -> CarHeuristic

+ `car_name` - The name of the car that want

Returns an instance of CarHeuristic.

### predict_car(self, car: CarHeuristic) -> dict

Evaluates the model that was passed in.

+ `car` - An instance of CarHeuristic

Returns a dictionary of keys (currently `may_ground_shot`, `may_jump_shot`, `may_double_jump_shot`, and `may_aerial`) that are either `True` or `False`.
