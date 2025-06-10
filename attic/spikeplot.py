#!/usr/bin/python3

'''
Spike-raster plotting program

Copyright (C) 2025 Simon D. Levy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, in version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

try:
    import serial
except Exception:
    serial = None

from sys import stdout
import socket
from time import sleep
import threading
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_serial(t, fig, lines, newspike):

    for row, line in enumerate(lines):
        ydata = line.get_ydata(line)
        ydata = np.roll(ydata, -1)
        ydata[-1] = 1 if row == newspike[0] else 0
        ydata = np.roll(ydata, -1)
        ydata[-1] = 0
        line.set_ydata(ydata)

    return lines


def serial_thread(port, fig, spiketrains, newspike):

    while True:

        # Quit on plot close
        if not plt.fignum_exists(fig.number):
            break

        newspike[0] = ord(port.read())

        sleep(0)  # yield to main thread


def client_thread(client, fig, spiketrains, n_neurons, connected):

    while True:

        # Quit on plot close
        if not plt.fignum_exists(fig.number):
            break

        msg = client.recv(n_neurons)

        counts = [count for count in msg]

        if len(counts) > 0:
            for s in spiketrains:
                s['count'] = counts[s['index']]
        else:
            connected[0] = False

        sleep(.001)  # yield


def animate_client(frame, spiketrains, ticks, showcounts, connected):

    for spiketrain in spiketrains:

        count = spiketrain['count']

        # Add count as a legend if indicated
        if showcounts:
            spiketrain['ax'].legend(['%d' % count], handlelength=0,
                                    loc='lower left',
                                    bbox_to_anchor=(0.01, 0.005))

        if connected[0] and count > 0:

            period = int(np.round(100 / count))

            lines = spiketrain['lines']

            # Add a new spike periodically
            if ticks[0] % period == 0:
                lines.append(spiketrain['ax'].plot((100, 100), (0, 1), 'k'))

            # Prune spikes as the move left outside the window
            if len(lines) > 0 and lines[0][0].get_xdata()[0] < 0:
                lines.pop(0)

            # Shift spikes to left
            for line in lines:
                ln = line[0]
                xdata = ln.get_xdata()
                ln.set_xdata(xdata - 1)

    ticks[0] += 1

    sleep(0.01)


def set_axis_properties(ax, neuron_ids, index, is_last):

    ax.set_ylim((0, 1.1))
    ax.set_xlim((0, 100))
    ax.set_ylabel(neuron_ids[index])
    ax.set_xticks([])
    ax.set_yticks([])

    if is_last:
        ax.set_xlabel('1 sec')


def load_neuron_aliases(filename):

    # Load network from JSON file
    network = json.loads(open(filename).read())

    # Get neuron aliases by sorting node ids from network JSON
    return sorted([int(node['id']) for node in network['Nodes']])


def run_socket(args, fig, axes, spiketrains, ticks, total_neurons):

    port = int(args.port)

    # Create a Bluetooth or IP socket depending on address format
    client = (socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,
              socket.BTPROTO_RFCOMM)
              if ':' in args.address
              else socket.socket(socket.AF_INET, socket.SOCK_STREAM))

    # Attempt to connect to the server until connection is made
    while True:
        try:
            client.connect((args.address, port))
            print('Connected to server')
            break
        except Exception:
            print('Waiting for server %s:%d to start' %
                  (args.address, port))
            sleep(1)

    connected = [True]

    # Start the client thread
    thread = threading.Thread(
            target=client_thread,
            args=(client, fig, spiketrains, total_neurons, connected))

    thread.start()

    # Start the animation thread
    anim = animation.FuncAnimation(
            fig, animate_client, interval=1,
            fargs=(spiketrains, ticks, args.display_counts, connected),
            blit=False, cache_frame_data=False)

    return anim


def run_serial(args, fig, axes, spiketrains):

    try:
        port = serial.Serial(args.port, 115200)

    except Exception:
        print('Unable to open connection to %s' % args.port)
        return

    newspike = [-1]  # Start with OOB neuron index

    # Start the data-acquisition thread
    thread = threading.Thread(
            target=serial_thread, args=(port, fig, spiketrains, newspike))

    thread.start()

    # Allow port to open
    sleep(0.25)

    x = np.arange(0, 100)
    y = np.zeros(100)
    lines = [ax.plot(x, y, 'k', animated=True)[0] for ax in axes]

    anim = animation.FuncAnimation(
            fig, animate_serial, interval=1,
            fargs=(fig, lines, newspike),
            blit=True, cache_frame_data=False)

    return anim


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--filename',
                       help='Name of JSON-formatted network file to load')
    group.add_argument('-n', '--neuron-count', type=int,
                       help='Number of neurons')
    parser.add_argument('-a', '--address', help='address (IP or MAC)')
    parser.add_argument('-p', '--port', help='port (number or device)',
                        required=True)
    parser.add_argument('-i', '--ids', help='neuron ids', default='all')
    parser.add_argument('-v', '--video', help='video file to save',
                        default=None)
    parser.add_argument('-d', '--display-counts', help='display counts',
                        action='store_true')
    parser.add_argument('-t', '--title', help='title', default='Spikes')
    args = parser.parse_args()

    neuron_aliases = (load_neuron_aliases(args.filename)
                      if args.filename is not None
                      else list(range(args.neuron_count)))

    # Get desired neuron IDs to plot from command line
    neuron_ids = (neuron_aliases
                  if args.ids == 'all'
                  else list(map(int, args.ids.strip().split(','))))

    # Bozo filter
    for neuron_id in neuron_ids:
        if neuron_id not in neuron_aliases:
            print('Neuron %d not in network; quitting' % neuron_id)
            exit(1)

    # Create figure and axes in which to plot spike trains
    fig, axes = plt.subplots(len(neuron_ids))

    anim = None

    # Multiple neurons
    if isinstance(axes, np.ndarray):

        for k, ax in enumerate(axes):

            set_axis_properties(ax, neuron_ids, k, k == len(axes)-1)

        # Make list of spike-train info
        spiketrains = [{'ax': ax, 'lines': [], 'count': 0,
                        'index': neuron_aliases.index(nid)}
                       for ax, nid in zip(axes, neuron_ids)]

    # Just one neuron
    else:

        set_axis_properties(axes, neuron_ids, 0, True)

        spiketrains = [{'ax': axes, 'lines': [], 'count': 0,
                        'index': neuron_aliases.index(neuron_ids[0])}]

    plt.suptitle(args.title)

    total_neurons = len(neuron_aliases)

    # Create timestep count, to be shared between threads
    ticks = [0]

    # Address and port; run as client
    if args.address is not None:
        anim = run_socket(args, fig, axes, spiketrains, ticks, total_neurons)

    # Just port; read directly from port
    elif serial is not None:
        anim = run_serial(args, fig, axes, spiketrains)

    # Oopsie!
    else:
        print('Need to install pyserial')
        exit(0)

    plt.show()

    # Save video file if indicated
    if args.video is not None:
        print(('Saving animation to ' + args.video), end=' ... ')
        stdout.flush()
        anim.save(args.video, writer=animation.FFMpegWriter(fps=30))
        print()


main()
