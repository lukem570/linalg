#include <cbuild/cbuild.hpp>

int build(CBuild::Context context) {
    
    CBuild::Executable test (
        context,
        "./test.cpp",
        "test"
    );

    test.includeDirectory("include");
    test.compile();
    
    return 0;
}