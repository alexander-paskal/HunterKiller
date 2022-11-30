from audioplayer import AudioPlayer
import time

class Speaker:

    def __init__(self):
        self.starting_time = time.time()
        self.mode = 0
        self.robot_dead = False
        self.looking_mode = AudioPlayer("music.mp3")
        self.hunting_mode = AudioPlayer("clown.mp3")
        self.explosion = AudioPlayer("explosion.mp3")

        self.play_music()

    def check_mode(self):
        if time.time() - self.starting_time > 5:
            self.mode += 1
            self.starting_time = time.time()
            return True
        else:
            return False

    def play_music(self):

        if self.mode == 0:
            self.hunting_mode.stop()
            self.explosion.stop()
            self.looking_mode.play(loop=True, block=False)
        elif self.mode == 1:
            self.looking_mode.stop()
            self.explosion.stop()
            self.hunting_mode.play(loop=True, block=False)
        elif self.mode == 2:
            self.hunting_mode.stop()
            self.looking_mode.stop()
            self.explosion.play(loop=False, block=True)
            self.robot_dead = True

        while True:
            if self.robot_dead:
                break
            if self.check_mode():
                self.play_music()


speaker = Speaker()