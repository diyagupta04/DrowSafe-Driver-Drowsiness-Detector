import pygame

pygame.mixer.init()
pygame.mixer.music.load("public/alarm.wav")  # or alarm.mp3
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    continue