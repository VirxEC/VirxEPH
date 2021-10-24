from traceback import print_exc

from rlbot.agents.base_script import BaseScript
from rlbot.utils.structures.game_data_struct import GameTickPacket

from VirxEPH import PacketHeuristics


class VirxEPHExternal(BaseScript):
    def __init__(self):
        super().__init__("VirxEPH-external")
        self.packet_heuristics = PacketHeuristics(verbose=True)

    def main(self):
        while 1:
            try:
                packet: GameTickPacket = self.wait_game_tick_packet()
                ball_prediction_struct = self.get_ball_prediction_struct()
                self.renderer.begin_rendering()
                
                # self.renderer.draw_polyline_3d(tuple(ball_slice.physics.location for ball_slice in ball_prediction_struct.slices[::10]), self.renderer.yellow())

                for zones in self.packet_heuristics.zones:
                    for zone in zones:
                        if zone is not None:
                            self.renderer.draw_line_3d(zone.min.copy(), zone.max.copy(), self.renderer.pink())

                added_packet = self.packet_heuristics.add_tick(packet, ball_prediction_struct)

                if not added_packet:
                    continue

                # render the predictions
                for i in range(packet.num_cars):
                    car = packet.game_cars[i]
                    car_name = self.packet_heuristics.get_true_car_name(car.name)

                    prediction_values = self.packet_heuristics.get_car(car.name)

                    if prediction_values is None:
                        print(f"Skipping {car.name} - ({car_name})")
                        continue

                    predictions = self.packet_heuristics.predict_car(prediction_values)
                    zone_id = self.packet_heuristics.car_tracker[car.name]['zone_id']
                    surrounding_zone_ids = self.packet_heuristics.get_surrounding_zone_ids(zone_id)
                    raw_predictions = "[" + ", ".join(str(round(prediction, 3)) for prediction in prediction_values) + "]"
                    true_predictions = "\n".join(key for key, value in predictions.items() if value)
                    self.renderer.draw_string_3d(car.physics.location, 1, 1, f"zone_id: {zone_id}\nsurrounding_zone_ids: {surrounding_zone_ids}\nraw_predictions: {raw_predictions}\n{true_predictions}", self.renderer.cyan())
            except Exception:
                print_exc()
            finally:
                self.renderer.end_rendering()

if __name__ == "__main__":
    VirxEPHExternal = VirxEPHExternal()
    VirxEPHExternal.main()
