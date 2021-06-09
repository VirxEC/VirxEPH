from traceback import print_exc

from rlbot.agents.base_script import BaseScript
from rlbot.utils.structures.game_data_struct import GameTickPacket

from VirxEPH import PacketHeuristicAnalyzer


class VirxEPHExternal(BaseScript):
    def __init__(self):
        super().__init__("VirxEPH-external")
        self.packet_heuristic = PacketHeuristicAnalyzer()

    def main(self):
        while 1:
            try:
                packet: GameTickPacket = self.wait_game_tick_packet()

                added_packet = self.packet_heuristic.add_packet(packet)

                if not added_packet:
                    continue

                # render the predictions
                for i in range(packet.num_cars):
                    car = packet.game_cars[i]
                    prediction_values = self.packet_heuristic.get_car(car.name, car.team)

                    if prediction_values is None:
                        print(f"Skipping {car.name}")
                        continue

                    predictions = self.packet_heuristic.predict_car(prediction_values)

                    raw_predictions = "[" + ", ".join(str(round(prediction, 2)) for prediction in prediction_values) + "]"
                    true_predictions = "\n".join(key for key, value in predictions.items() if value)
                    self.renderer.begin_rendering()
                    self.renderer.draw_string_3d(car.physics.location, 1, 1, raw_predictions + "\n" + true_predictions, self.renderer.purple())
                    self.renderer.end_rendering()
            except Exception:
                print_exc()

if __name__ == "__main__":
    VirxEPHExternal = VirxEPHExternal()
    VirxEPHExternal.main()
