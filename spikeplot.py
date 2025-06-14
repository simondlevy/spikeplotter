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

from sys import stdout
from struct import unpack
import socket
from time import sleep
import threading
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def threadfun(client, fig, spiketrains, n_neurons, connected, logarithmic):

    while True:

        # Quit on plot close
        if not plt.fignum_exists(fig.number):
            break

        counts = unpack('I'*n_neurons, client.recv(4*n_neurons))

        if len(counts) > 0:
            for k,s in enumerate(spiketrains):
                s['count'] = counts[k]
        else:
            connected[0] = False

        sleep(.001)  # yield


def animfun(frame, spiketrains, ticks, showvals, connected, logarithmic,
            time):

    for spiketrain in spiketrains:

        rawcount = spiketrain['count'] 

        count = np.log(rawcount+1) if logarithmic else rawcount

        # Add count as a legend if indicated
        if showvals:
            spiketrain['ax'].legend(['%d' % rawcount], handlelength=0,
                                    loc='lower left',
                                    bbox_to_anchor=(0.01, 0.005))

        if connected[0] and count > 0:

            period = int(np.round((100000/time) / count))

            lines = spiketrain['lines']

            # If spikes are coming faster than we can plot them, use a thick
            # line
            lw = 20 if period == 0 else 1

            # Otherwise add a new line periodically
            if period == 0 or ticks[0] % period == 0:
                lines.append(spiketrain['ax'].plot((100, 100), (0, 1), 'k',
                                                   linewidth=lw))

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


def make_axis(ax, neuron_ids, index, time, logarithmic, is_last):

    ax.set_ylim((0, 1.1))
    ax.set_xlim((0, 100))
    nid = neuron_ids[index]
    ax.set_ylabel('log(%s)' % nid if logarithmic else nid)
    ax.set_xticks([])
    ax.set_yticks([])

    if is_last:
        ax.set_xlabel('%d msec' % time)


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--address', help='address (IP or MAC)',
                        required=True)
    parser.add_argument('-p', '--port', help='port', type=int, required=True)
    parser.add_argument('-i', '--ids', help='neuron ids')
    parser.add_argument('-v', '--video', help='video file to save',
                        default=None)
    parser.add_argument('-d', '--display-counts', help='display counts',
                        action='store_true')
    parser.add_argument('-l', '--logarithmic', help='use logarithm of counts',
                        action='store_true')
    parser.add_argument('-t', '--time', type=int, default=1000,
                        help='Time span in milliseconds')
    args = parser.parse_args()

    # Get desired neuron IDs to plot from command line
    neuron_ids = args.ids.strip().split(',')

    # Create figure and axes in which to plot spike trains
    fig, axes = plt.subplots(len(neuron_ids))

    # Multiple neurons
    if isinstance(axes, np.ndarray):

        for k, ax in enumerate(axes):

            make_axis(ax, neuron_ids, k, args.time, args.logarithmic,
                      k == len(axes)-1)

        # Make list of spike-train info
        spiketrains = [{'ax': ax, 'lines': [], 'count': 0}
                       for ax, nid in zip(axes, neuron_ids)]

    # Just one neuron
    else:

        make_axis(axes, neuron_ids, 0, args.logarithmic, True)

        spiketrains = [{'ax': axes, 'lines': [], 'count': 0}]

    # Create timestep count, to be shared between threads
    ticks = [0]

    # Create a Bluetooth or IP socket depending on address format
    client = (socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,
              socket.BTPROTO_RFCOMM)
              if ':' in args.address
              else socket.socket(socket.AF_INET, socket.SOCK_STREAM))

    # Attempt to connect to the server until connection is made
    while True:
        try:
            client.connect((args.address, args.port))
            print('Connected to server')
            break
        except Exception:
            print('Waiting for server %s:%d to start' %
                  (args.address, args.port))
            sleep(1)

    connected = [True]

    # Start the client thread
    thread = threading.Thread(
            target=threadfun,
            args=(client, fig, spiketrains, len(neuron_ids), connected,
                  args.logarithmic))
    thread.start()

    # Star the animation thread
    ani = animation.FuncAnimation(
            fig=fig,
            func=animfun,
            fargs=(spiketrains, ticks, args.display_counts, connected,
                   args.logarithmic, args.time),
            cache_frame_data=False,
            interval=1)

    plt.show()

    # Save video file if indicated
    if args.video is not None:
        print(('Saving animation to ' + args.video), end=' ... ')
        stdout.flush()
        ani.save(args.video, writer=animation.FFMpegWriter(fps=30))
        print()


main()
