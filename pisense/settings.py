# vim: set et sw=4 sts=4 fileencoding=utf-8:
#
# Alternative API for the Sense HAT
# Copyright (c) 2016-2018 Dave Jones <dave@waveform.org.uk>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Defines the :class:`SenseSettings` class representing the Sense HAT's
calibration settings.
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
)

try:
    import RTIMU
except ImportError:
    RTIMU = None

class SenseSettings(object):
    """
    Represents the calibration settings for the Sense HAT.

    The *settings_file* refers to the INI-style file containing all calibration
    settings for the Sense HAT. For no particularly good reason, the underlying
    library requires that this filename ends with '.ini'.

    .. warning::

        If the specified file does not exist, it will be created with default
        calibration settings. Hence you *should* ensure that the location
        specified either exists or is writeable by the current user.
        Furthermore, if the file successfully loads the underlying library
        will attempt to overwrite it with "cleaned" values. If you wish to
        keep modifications to the file (comments, etc.) ensure that the file is
        effectively read-only for the executing user.

        Yes, this is a ridiculous requirement and while I would dearly love to
        re-write this chunk of the underlying library, it's not something I
        have time for currently!
    """
    # pylint: disable=too-few-public-methods

    __slots__ = ('_settings',)

    def __init__(self, settings_file=None, emulate=False):
        if emulate:
            from sense_emu import RTIMU
        else:
            import RTIMU
        if settings_file is None:
            settings_file = '/etc/RTIMULib.ini'
        if not settings_file.endswith('.ini'):
            raise ValueError("RTIMULib doesn't accept settings filenames "
                             "without the .ini extension; yes, it's dumb")
        # Actually, it's worse than that; it silently adds .ini to whatever you
        # specify. So, the settings filename you specify isn't actually what
        # gets used! Worse still, if it decides the filename is too long (200
        # characters for some arbitrary reason ... not that that's a limit on
        # any platform I'm aware of) it'll silently continue with its default
        # filename, or just *print* an error if the library's been compiled
        # with the appropriate option, but either way your code will be happily
        # oblivious to the fact it's not using the requested settings. Oh well!
        #
        # RANT TIME
        #
        # But we're not finished. Now we come to the staggeringly dumb bit.
        # Loading (which happens implicitly upon construction) goes something
        # like this:
        #
        # 1. Load a bunch of default constants. Okay, that bit's sensible.
        #
        # 2. Attempt to open the file. Not found? Not accessible? Can't open it
        #    the file for any reason? Never mind, let's try and save the
        #    defaults! Ignoring the fact the default location is root writable
        #    only, the disk might be full, etc. etc. Incidentally, return true
        #    or false to indicate whether saving succeeded. But then ignore
        #    that and return None anyway.
        #
        # 3. So, you managed to open the settings file? Great, let's start
        #    reading it. Oh ... there's an error on this line? Silently close
        #    the file and return false ... then turn that into None anyway.
        #    So you've no idea that you just loaded partial settings because
        #    the file is corrupt.
        #
        # 4. Well done! You've managed to parse the whole settings file. What's
        #    that? You don't trust the file-system? You want to silently try &
        #    re-write the entire settings file to "make sure settings file is
        #    correct and complete" (seriously, that's a comment in the code).
        #    Never mind that you just read it successfully. Or that there might
        #    be commented sections that the user wants to keep. Or that the
        #    disk might be full. No, sod that, let's re-write it all anyway.
        #    And then return true. Which we'll ignore and change into None.
        #
        # Seriously though, the settings code in RTIMULib is a *masterpiece* of
        # idiocy. I haven't even covered how all the saving code ignores
        # fprintf's return values... Anyway, rant over.
        self._settings = RTIMU.Settings(settings_file[:-4])

    @property
    def settings(self):
        """
        Returns the underlying RTIMULib settings object.
        """
        return self._settings
