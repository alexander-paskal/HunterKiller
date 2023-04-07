from audioplayer import AudioPlayer
import time

class Speaker:

    def __init__(self):
        self.starting_time = time.time()
        #self.robot_dead = False
        self.looking_mode = AudioPlayer("music.mp3")
        self.hunting_mode = AudioPlayer("clown.mp3")
        self.explosion = AudioPlayer("explosion.mp3")
        self.hoya = AudioPlayer("hoya.mp3")

    def play_music(self, music):

        if music == "music.mp3":
            self.hunting_mode.stop()
            self.explosion.stop()
            self.hoya.stop()
            self.looking_mode.play(loop=True, block=False)
        elif music == "clown.mp3":
            self.looking_mode.stop()
            self.explosion.stop()
            self.hoya.stop()
            self.hunting_mode.play(loop=True, block=False)
        elif music == "explosion.mp3":
            self.hunting_mode.stop()
            self.looking_mode.stop()
            self.explosion.play(loop=False, block=True)
            self.hoya.stop()
            self.robot_dead = True

        elif music == "hoya.mp3":
            self.hunting_mode.stop()
            self.looking_mode.stop()
            self.explosion.stop()
            self.hoya.play(loop= True, block= False)
