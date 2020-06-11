### 概要
状況: 噂のstreamlitをinstallしようとしたらエラー
環境: macOS Catalina 10.15.4
解決方法: ファイルのリネーム (sudo mv /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk.bak)


### どんなエラーか？

インストール中に以下のようなエラー文が出現する。
watchdogのinstallでつまづくらしい。


```
    Running setup.py install for watchdog ... error
    ERROR: Command errored out with exit status 1:
     command: /Users/{user}/.pyenv/versions/3.6.9/bin/python3.6 -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/private/var/folders/4f/v1bm8p692yv8lclpdt29vtlc0000gn/T/pip-install-1fu4waam/watchdog/setup.py'"'"'; __file__='"'"'/private/var/folders/4f/v1bm8p692yv8lclpdt29vtlc0000gn/T/pip-install-1fu4waam/watchdog/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record /private/var/folders/4f/v1bm8p692yv8lclpdt29vtlc0000gn/T/pip-record-okwa84ro/install-record.txt --single-version-externally-managed --compile --install-headers /Users/{user}/.pyenv/versions/3.6.9/include/python3.6m/watchdog
         cwd: /private/var/folders/4f/v1bm8p692yv8lclpdt29vtlc0000gn/T/pip-install-1fu4waam/watchdog/
    Complete output (299 lines):
    running install
    running build
    running build_py
    creating build
    creating build/lib.macosx-10.14-x86_64-3.6
    creating build/lib.macosx-10.14-x86_64-3.6/watchdog
    copying src/watchdog/watchmedo.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog
    copying src/watchdog/version.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog
    copying src/watchdog/events.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog
    copying src/watchdog/__init__.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog
    creating build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/unicode_paths.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/compat.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/win32stat.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/__init__.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/dirsnapshot.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/delayed_queue.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/platform.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/bricks.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    copying src/watchdog/utils/echo.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/utils
    creating build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/fsevents.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/inotify.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/__init__.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/api.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/inotify_buffer.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/winapi.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/read_directory_changes.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/kqueue.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/inotify_c.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/polling.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    copying src/watchdog/observers/fsevents2.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/observers
    creating build/lib.macosx-10.14-x86_64-3.6/watchdog/tricks
    copying src/watchdog/tricks/__init__.py -> build/lib.macosx-10.14-x86_64-3.6/watchdog/tricks
    running egg_info
    writing src/watchdog.egg-info/PKG-INFO
    writing dependency_links to src/watchdog.egg-info/dependency_links.txt
    writing entry points to src/watchdog.egg-info/entry_points.txt
    writing requirements to src/watchdog.egg-info/requires.txt
    writing top-level names to src/watchdog.egg-info/top_level.txt
    reading manifest file 'src/watchdog.egg-info/SOURCES.txt'
    reading manifest template 'MANIFEST.in'
    warning: no files found matching '*.h' under directory 'src'
    writing manifest file 'src/watchdog.egg-info/SOURCES.txt'
    running build_ext
    building '_watchdog_fsevents' extension
    creating build/temp.macosx-10.14-x86_64-3.6
    creating build/temp.macosx-10.14-x86_64-3.6/src
    clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -I/Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/usr/include -I/Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/usr/include -DWATCHDOG_VERSION_STRING="0.10.2" -DWATCHDOG_VERSION_MAJOR=0 -DWATCHDOG_VERSION_MINOR=10 -DWATCHDOG_VERSION_BUILD=2 -I/Users/{user}/.pyenv/versions/3.6.9/include/python3.6m -c src/watchdog_fsevents.c -o build/temp.macosx-10.14-x86_64-3.6/src/watchdog_fsevents.o -std=c99 -pedantic -Wall -Wextra -fPIC -Wno-nullability-completeness -Wno-nullability-extension -Wno-newline-eof -Wno-error=unused-command-line-argument
    In file included from src/watchdog_fsevents.c:24:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Headers/CoreServices.h:39:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Headers/LaunchServices.h:23:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Headers/IconsCore.h:23:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/OSServices.framework/Headers/OSServices.h:29:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/OSServices.framework/Headers/CSIdentity.h:26:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/OSServices.framework/Headers/CSIdentityBase.h:23:
    /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Security.framework/Headers/SecBase.h:142:74: error: expected ','
    typedef struct CF_BRIDGED_TYPE(id) __SecKeychainItem *SecKeychainItemRef API_UNAVAILABLE(ios, watchos, tvos, macCatalyst);
                                                                             ^
    /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/usr/include/os/availability.h:93:114: note: expanded from macro 'API_UNAVAILABLE'
        #define API_UNAVAILABLE(...) __API_UNAVAILABLE_GET_MACRO(__VA_ARGS__,__API_UNAVAILABLE6, __API_UNAVAILABLE5, __API_UNAVAILABLE4,__API_UNAVAILABLE3,__API_UNAVAILABLE2,__API_UNAVAILABLE1, 0)(__VA_ARGS__)
                                                                                                                     ^
    In file included from src/watchdog_fsevents.c:24:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Headers/CoreServices.h:39:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Headers/LaunchServices.h:23:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Headers/IconsCore.h:23:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/OSServices.framework/Headers/OSServices.h:29:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/OSServices.framework/Headers/CSIdentity.h:26:
    In file included from /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreServices.framework/Frameworks/OSServices.framework/Headers/CSIdentityBase.h:23:
    /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Security.framework/Headers/SecBase.h:148:78: error: expected ','
    typedef struct CF_BRIDGED_TYPE(id) __SecKeychainSearch *SecKeychainSearchRef API_UNAVAILABLE(ios, watchos, tvos, macCatalyst);
                                                                                 ^
    /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/usr/include/os/availability.h:93:114: note: expanded from macro 'API_UNAVAILABLE'
        #define API_UNAVAILABLE(...) __API_UNAVAILABLE_GET_MACRO(__VA_ARGS__,__API_UNAVAILABLE6, __API_UNAVAILABLE5, __API_UNAVAILABLE4,__API_UNAVAILABLE3,__API_UNAVAILABLE2,__API_UNAVAILABLE1, 0)(__VA_ARGS__)
                                                                                                                     ^
中略
    fatal error: too many errors emitted, stopping now [-ferror-limit=]
    20 errors generated.
    error: command 'clang' failed with exit status 1
    ----------------------------------------
ERROR: Command errored out with exit status 1: /Users/{user}/.pyenv/versions/3.6.9/bin/python3.6 -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/private/var/folders/4f/v1bm8p692yv8lclpdt29vtlc0000gn/T/pip-install-1fu4waam/watchdog/setup.py'"'"'; __file__='"'"'/private/var/folders/4f/v1bm8p692yv8lclpdt29vtlc0000gn/T/pip-install-1fu4waam/watchdog/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' install --record /private/var/folders/4f/v1bm8p692yv8lclpdt29vtlc0000gn/T/pip-record-okwa84ro/install-record.txt --single-version-externally-managed --compile --install-headers /Users/{user}/.pyenv/versions/3.6.9/include/python3.6m/watchdog Check the logs for full command output.
```

### 解決方法

watchdogのissueで`/Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk`を削除すると良いとある。

https://github.com/gorakhargosh/watchdog/issues/628#issuecomment-581480649


この方法以外にも、streamlitのissueでいろいろ議論がされている(こちらではcondaを使うと早いという話)。
https://github.com/streamlit/streamlit/issues/283

流石に`/Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk`を消すのは怖かったので、リネームすることで解決した。

つまり
```
pip install streamlit
```

