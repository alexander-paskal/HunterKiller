from audioplayer import AudioPlayer
import time

class Speaker:

    def __init__(self):
        self.starting_time = time.time()
        #self.robot_dead = False
        self.looking_mode = AudioPlayer("music.mp3")
        self.hunting_mode = AudioPlayer("clown.mp3")
        self.explosion = AudioPlayer("explosion.mp3")

        self.play_music()

    def play_music(self, music):

        if music == "music.mp3":
            self.hunting_mode.stop()
            self.explosion.stop()
            self.looking_mode.play(loop=True, block=False)
        elif music == "clown.mp3":
            self.looking_mode.stop()
            self.explosion.stop()
            self.hunting_mode.play(loop=True, block=False)
        elif music == "explision.mp3":
            self.hunting_mode.stop()
            self.looking_mode.stop()
            self.explosion.play(loop=False, block=True)
            self.robot_dead = True

speaker = Speaker()
