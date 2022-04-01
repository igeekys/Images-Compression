if(NOT EXISTS "/mnt/md0p1/GPH/AAAA/libbpg-0.9.8/x265.out/8bit/install_manifest.txt")
    message(FATAL_ERROR "Cannot find install manifest: '/mnt/md0p1/GPH/AAAA/libbpg-0.9.8/x265.out/8bit/install_manifest.txt'")
endif()

file(READ "/mnt/md0p1/GPH/AAAA/libbpg-0.9.8/x265.out/8bit/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
    message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
    if(EXISTS "$ENV{DESTDIR}${file}" OR IS_SYMLINK "$ENV{DESTDIR}${file}")
        exec_program("/usr/bin/cmake" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
                     OUTPUT_VARIABLE rm_out
                     RETURN_VALUE rm_retval)
        if(NOT "${rm_retval}" STREQUAL 0)
            message(FATAL_ERROR "Problem when removing '$ENV{DESTDIR}${file}'")
        endif(NOT "${rm_retval}" STREQUAL 0)
    else()
        message(STATUS "File '$ENV{DESTDIR}${file}' does not exist.")
    endif()
endforeach(file)
