#!/usr/bin/env python3

import Adafruit_PCA9685
import time


try:

    pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)

    pwm.set_pwm_freq(60)

    def set_servo_angle(channel, angle):
        pulse = int(4096 * ((angle * 11) + 500) / 20000)
        pwm.set_pwm(channel, 0, pulse)
        print(f"Channel {channel} set to {angle}°")


    print("Centering servos...")
    set_servo_angle(2, 60)
    set_servo_angle(3, 90)
    time.sleep(1)
    print("Done!")

except Exception as e:
    print(f"Error: {str(e)}")
    raise
