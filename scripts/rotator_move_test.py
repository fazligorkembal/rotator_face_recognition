import argparse
import glob
import time

try:
    import serial
except ImportError:
    serial = None


def find_serial_port():
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


def send_rad_command(motor1_rad_s, motor2_rad_s, port=None, baudrate=115200):
    if serial is None:
        raise SystemExit("pyserial kurulu degil. Kurulum: pip3 install pyserial")

    serial_port = port or find_serial_port()
    if not serial_port:
        raise SystemExit("Arduino portu bulunamadi. --port ile belirt.")

    command = f"rad {motor1_rad_s} {motor2_rad_s}\n"

    with serial.Serial(serial_port, baudrate, timeout=1) as ser:
        time.sleep(2.0)  # Arduino reset olursa portun toparlanmasini bekle
        ser.write(command.encode("utf-8"))
        ser.flush()
        print(f"Gonderildi -> {serial_port}: {command.strip()}")

        response = ser.readline().decode("utf-8", errors="ignore").strip()
        if response:
            print(f"Arduino -> {response}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motor1", type=float, help="Motor1 hizi (rad/s)")
    parser.add_argument("motor2", type=float, help="Motor2 hizi (rad/s)")
    parser.add_argument("--port", default=None, help="Orn: /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    args = parser.parse_args()

    send_rad_command(args.motor1, args.motor2, args.port, args.baudrate)


if __name__ == "__main__":
    main()
