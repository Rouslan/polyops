#include <vector>
#include <iomanip>

#ifdef HAVE_GMP_LIBRARY
#include <gmpxx.h>
#include <random>
#include <utility>
#endif

#define BOOST_UT_DISABLE_MODULE
#include "third_party/boost/ut.hpp"

#include "../include/poly_ops/large_ints.hpp"

using namespace boost::ut;

#ifdef HAVE_GMP_LIBRARY
template<unsigned int N,bool Signed> mpz_class to_gmp(const poly_ops::large_ints::compound_xint<N,Signed> &x) {
    mpz_class r;
    if constexpr(Signed) {
        if(x.negative()) {
            mpz_import(r.get_mpz_t(),N,-1,sizeof(poly_ops::large_ints::full_int),0,0,(-x).data());
            return -r;
        }
    }

    mpz_import(r.get_mpz_t(),N,-1,sizeof(poly_ops::large_ints::full_int),0,0,x.data());
    return r;
}

template<unsigned int N,bool Signed> poly_ops::large_ints::compound_xint<N,Signed> from_gmp(const mpz_class &x) {
    using namespace poly_ops::large_ints;

    if(Signed && sgn(x) < 0) throw std::logic_error("cannot store negative value in unsigned class");

    compound_xint<N,Signed> r{0};
    std::size_t bitsize = sizeof(full_int)*8;
    std::size_t words = (mpz_sizeinbase(x.get_mpz_t(),2) + bitsize - 1 + Signed) / bitsize;
    if(words > N) throw std::logic_error("input value is too big");
    mpz_export(r.data(),nullptr,-1,sizeof(full_int),0,0,x.get_mpz_t());
    if constexpr(Signed) {
        if(sgn(x)) return -r;
    }
    return r;
}
#endif

namespace poly_ops::large_ints {
    namespace detail {
        template<unsigned int N> std::ostream &output_tail(std::ostream &os,const_compound_xint_ref<N> x) {
            os << ',' << std::setw(sizeof(full_int)*2) << x.hi();
            if constexpr(N > 1) {
                return output_tail(os,x.lo());
            } else {
                return os;
            }
        };
        template<unsigned int N,bool Signed> std::ostream &output_head(std::ostream &os,const compound_xint<N,Signed> &x) {
            if constexpr(N > 1) {
                return output_tail(os << x.hi(),x.lo());
            } else {
                return os << x.hi();
            }
        };
    }

    template<unsigned int N,bool Signed> std::ostream &operator<<(std::ostream &os,const compound_xint<N,Signed> &x) {
        if constexpr(Signed) {
            if(x.negative()) {
                return detail::output_head(os << "-0x" << std::hex << std::setfill('0'),-x);
            }
        }
        return detail::output_head(os << "0x" << std::hex << std::setfill('0'),x);
        //return os << to_gmp(x);
    };
}

#ifdef HAVE_GMP_LIBRARY
template<unsigned int N,bool Signed> bool operator==(const poly_ops::large_ints::compound_xint<N,Signed> &a,const mpz_class &b) {
    return to_gmp(a) == b;
}

template<unsigned int N> using compound_int_pair = std::pair<poly_ops::large_ints::compound_int<N>,poly_ops::large_ints::compound_int<N>>;

/* Only the bottom N-1 parts will be random. The top part will be the sign extension. */
template<unsigned int N> std::vector<compound_int_pair<N>> gen_random_compound_int_pairs() {
    using namespace poly_ops::large_ints;

    static_assert(N > 1);

    constexpr int test_run_count = 1000;

    std::default_random_engine re(std::random_device{}());
    std::uniform_int_distribution<full_uint> dist;

    auto rand_compound_int = [&] {
        compound_int<N> x;
        for(unsigned int i=0; i<N-1; ++i) x[i] = dist(re);
        x[N-1] = x[N-2] >> (sizeof(full_uint)*8-1);
        return x;
    };

    std::vector<std::pair<compound_int<4>,compound_int<4>>> test_input;
    test_input.reserve(test_run_count);
    for(int i=0; i<test_run_count; ++i) {
        test_input.emplace_back(rand_compound_int(),rand_compound_int());
    }
    return test_input;
}
#endif

constexpr inline poly_ops::large_ints::full_uint ONES = ~poly_ops::large_ints::full_uint(0);

int main() {
    using namespace boost::ut;
    using namespace poly_ops::large_ints;

    "Test compare"_test = [] {
        expect(boost::ut::eq(compound_int<1>(0),full_int(0)));
        expect(neq(compound_int<1>(1), full_int(0)));
        expect(boost::ut::eq(compound_int<2>(0), full_int(0)));
        expect(neq(compound_int<2>(1), full_int(0)));
        expect(neq(compound_int<2>(1,0u), full_int(0)));
        expect(neq(0xffffffffffffffff_compi, -1));
        expect(neq(compound_int<1>(ONES), compound_uint<1>(ONES)));
        expect(boost::ut::eq(compound_int<2>(0,ONES), compound_uint<1>(ONES)));
        expect(boost::ut::eq(0xfedcba9876543210fedcba9876543210_compi, 0xfedcba9876543210fedcba9876543210_compi));
        expect(le(0xfedcba9876543210fedcba9876543210_compi,0xfedcba9876543210fedcba9876543210_compi));
        expect(ge(0xfedcba9876543210fedcba9876543210_compi,0xfedcba9876543210fedcba9876543210_compi));
        expect(lt(0xfedcba9876543210fedcba9876543210_compi,0xfedcba9876543210fedcba9876543211_compi));
        expect(lt(-0xfedcba9876543210fedcba9876543210_compi,0xfedcba9876543210fedcba9876543210_compi));
        expect(le(-0xfedcba9876543210fedcba9876543210_compi, 0xfedcba9876543210fedcba9876543210_compi));
        expect(gt(0xfedcba9876543210fedcba9876543210_compi, -0xfedcba9876543210fedcba9876543210_compi));
        expect(ge(0xfedcba9876543210fedcba9876543210_compi, -0xfedcba9876543210fedcba9876543210_compi));
        expect(lt(compound_int<2>(-3751), compound_int<2>(-1260)));
    };

    "Test add"_test = [] {
        expect(boost::ut::eq(compound_int<1>(0) + compound_int<1>(0), compound_int<1>(0)));
        expect(boost::ut::eq(compound_int<1>(1) + compound_int<1>(0), compound_int<1>(1)));
        expect(boost::ut::eq(compound_int<1>(4) + compound_int<1>(-8), compound_int<1>(-4)));
        expect(boost::ut::eq(compound_int<2>(0,ONES) + full_int(1), compound_int<2>(1,0u)));
        expect(boost::ut::eq(compound_int<2>(ONES,ONES) + full_int(1), compound_int<2>(0,0u)));
        expect(boost::ut::eq(compound_int<2>(1,0u) + full_int(-1), compound_int<2>(0,ONES)));
        expect(boost::ut::eq(0xfedcba9876543210_compi + 0x1122334455667788_compi, 0x10ffeeddccbbaa998_compi));
        expect(boost::ut::eq(compound_uint<1>(ONES) + compound_uint<2>(1,0u), compound_uint<2>(1,ONES)));
        expect(boost::ut::eq(compound_int<1>(ONES) + compound_uint<2>(1,0u), compound_int<2>(0,ONES)));
    };

    "Test subtract"_test = [] {
        expect(boost::ut::eq(compound_int<1>(0) - compound_int<1>(0), compound_int<1>(0)));
        expect(boost::ut::eq(compound_int<1>(1) - compound_int<1>(0), compound_int<1>(1)));
        expect(boost::ut::eq(compound_int<1>(4) - compound_int<1>(-8), compound_int<1>(12)));
        expect(boost::ut::eq(compound_int<2>(0,ONES) - full_int(-1), compound_int<2>(1,0u)));
        expect(boost::ut::eq(compound_int<2>(ONES,ONES) - full_int(-1), compound_int<2>(0,0u)));
        expect(boost::ut::eq(compound_int<2>(1,0u) - full_int(1), compound_int<2>(0,ONES)));
        expect(boost::ut::eq(0xfedcba9876543210_compi - 0x1122334455667788_compi, 0xedba875420edba88_compi));
    };

    "Test multiply"_test = [] {
        expect(boost::ut::eq(mul(full_int(5),full_int(5)), 25));
        expect(boost::ut::eq(0xffffffff_compi * 5, 0x4fffffffb_compi));
        expect(boost::ut::eq(0xfffffffffffffff_compi * 5, 0x4ffffffffffffffb_compi));
        expect(boost::ut::eq(0xfffffffffffffff_compi * -9, -0x8ffffffffffffff7_compi));
        expect(boost::ut::eq(-0xfffffffffffffff_compi * 9, -0x8ffffffffffffff7_compi));
        expect(boost::ut::eq(0xffffffffffffffff_compi * 5, 0x4fffffffffffffffb_compi));
        expect(boost::ut::eq(0xffffffffffffffff_compi * -9, -0x8fffffffffffffff7_compi));
        expect(boost::ut::eq(0xfedcba9876543210fedcba9876543210_compi * 0xabc, 0xaafc962fc962fc96e6fc962fc962fc963c0_compi));
        expect(boost::ut::eq(0xfedcba9876543210fedcba9876543210_compi * -0xabc, -0xaafc962fc962fc96e6fc962fc962fc963c0_compi));
        expect(boost::ut::eq(-0xfedcba9876543210fedcba9876543210_compi * 0xabc, -0xaafc962fc962fc96e6fc962fc962fc963c0_compi));
        expect(boost::ut::eq(-0xfedcba9876543210fedcba9876543210_compi * -0xabc, 0xaafc962fc962fc96e6fc962fc962fc963c0_compi));
        expect(boost::ut::eq(0xfedcba9876543210fedcba9876543210_compi * 0x123456789abcdef0123456789abcdef_compi, 0x121fa00ad77d742247acc9140513b74458fab20783af1222236d88fe5618cf0_compi));
    };

    "Test shift"_test = [] {
        expect(boost::ut::eq((0xffffffff_compi >> 1), 0x7fffffff_compi));
        expect(boost::ut::eq((0xfffffffffffffff_compi >> 6), 0x3fffffffffffff_compi));
        expect(boost::ut::eq((0x00fffffffffffffff_compi << 6), 0x3ffffffffffffffc0_compi));
        expect(boost::ut::eq((0xfedcba9876543210fedcba9876543210_compi >> 13), 0x7f6e5d4c3b2a19087f6e5d4c3b2a1_compi));
        expect(boost::ut::eq((0xfedcba9876543210fedcba9876543210_compi >> 0), 0xfedcba9876543210fedcba9876543210_compi));
        expect(boost::ut::eq((0xfedcba9876543210fedcba9876543210_compi << 0), 0xfedcba9876543210fedcba9876543210_compi));
        expect(boost::ut::eq((0x0000fedcba9876543210fedcba9876543210_compi << 13), 0x1fdb97530eca86421fdb97530eca86420000_compi));
    };

    "Test divide"_test = [] {
        expect(boost::ut::eq(0xffffffff_compi / 11, 0x1745d174_compi));
        expect(boost::ut::eq(0xfffffffffffffff_compi / 12, 0x155555555555555_compi));
        expect(boost::ut::eq(0xfedcba9876543210fedcba9876543210_compi / 13, 0x139ad346ce067a014eae8481e1b7b514_compi));
        expect(boost::ut::eq(0xfedcba9876543210fedcba9876543210_compi / -13, -0x139ad346ce067a014eae8481e1b7b514_compi));
        expect(boost::ut::eq(-0xfedcba9876543210fedcba9876543210_compi / 13, -0x139ad346ce067a014eae8481e1b7b514_compi));
        expect(boost::ut::eq(-0xfedcba9876543210fedcba9876543210_compi / -13, 0x139ad346ce067a014eae8481e1b7b514_compi));
        expect(boost::ut::eq(divmod(0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).quot, 0x139ad346ce067a014eae8481e1b7b514_compi));
        expect(boost::ut::eq(divmod(0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).quot, -0x139ad346ce067a014eae8481e1b7b515_compi));
        expect(boost::ut::eq(divmod(-0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).quot, -0x139ad346ce067a014eae8481e1b7b515_compi));
        expect(boost::ut::eq(divmod(-0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).quot, 0x139ad346ce067a014eae8481e1b7b514_compi));
        expect(boost::ut::eq(0x121fa00ad77d742247acc9140513b74458fab20783af1222236d88fe5618cf0_compi / 0x123456789abcdef0123456789abcdef_compi, 0xfedcba9876543210fedcba9876543210_compi));
        expect(boost::ut::eq(0x121fa00ad77d742247acc9140513b74458fab20783af1222236d88fe5618cf0_compi / 0xfedcba9876543210fedcba9876543210_compi, 0x123456789abcdef0123456789abcdef_compi));
        expect(boost::ut::eq(0x1cca5d15d93cf3491bb81cfa68083b2d9ec8581cc3523a4c9_compi / 0x1a1c47c53b3211c279541261e94b415bdf18fc584762d7a72_compi, 1));
    };

    "Test modulo"_test = [] {
        expect(boost::ut::eq(divmod(0xfedcba9876543210fedcba9876543210_compi,13).rem, 12));
        expect(boost::ut::eq(divmod(0xfedcba9876543210fedcba9876543210_compi,-13).rem, 12));
        expect(boost::ut::eq(divmod(-0xfedcba9876543210fedcba9876543210_compi,13).rem, -12));
        expect(boost::ut::eq(divmod(-0xfedcba9876543210fedcba9876543210_compi,-13).rem, -12));
        expect(boost::ut::eq(divmod(0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).rem, 12));
        expect(boost::ut::eq(divmod(0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).rem, 1));
        expect(boost::ut::eq(divmod(-0xfedcba9876543210fedcba9876543210_compi,13,modulo_t::euclid).rem, 1));
        expect(boost::ut::eq(divmod(-0xfedcba9876543210fedcba9876543210_compi,-13,modulo_t::euclid).rem, 12));
        expect(boost::ut::eq(divmod(27,10,modulo_t::euclid).rem, 7));
        expect(boost::ut::eq(divmod(27,-10,modulo_t::euclid).rem, 3));
        expect(boost::ut::eq(divmod(-27,10,modulo_t::euclid).rem, 3));
        expect(boost::ut::eq(divmod(-27,-10,modulo_t::euclid).rem, 7));
        expect(boost::ut::eq(unmul<1>(27,10,modulo_t::euclid).rem, 7));
        expect(boost::ut::eq(unmul<1>(27,-10,modulo_t::euclid).rem, 3));
        expect(boost::ut::eq(unmul<1>(-27,10,modulo_t::euclid).rem, 3));
        expect(boost::ut::eq(unmul<1>(-27,-10,modulo_t::euclid).rem, 7));
    };

#ifdef HAVE_GMP_LIBRARY
    "Ops on random numbers"_test = [](const compound_int_pair<4> &values) {
        auto& [a,b] = values;

        auto a_gmp = to_gmp(a);
        auto b_gmp = to_gmp(b);

        expect(boost::ut::eq(a + b, a_gmp + b_gmp));
        expect(boost::ut::eq(a - b, a_gmp - b_gmp));
        expect(boost::ut::eq(a * b, a_gmp * b_gmp));
        if(b) expect(boost::ut::eq(a / b, a_gmp / b_gmp));
        expect(boost::ut::eq(a << 3, a_gmp << 3));
        expect(boost::ut::eq(a >> 100, a_gmp >> 100));
        expect(boost::ut::eq(a < b, a_gmp < b_gmp));
        expect(boost::ut::eq(a <= b, a_gmp <= b_gmp));
        expect(boost::ut::eq(a > b, a_gmp > b_gmp));
        expect(boost::ut::eq(a >= b, a_gmp >= b_gmp));
        expect(boost::ut::eq(poly_ops::large_ints::eq(a,b), a_gmp == b_gmp));
        expect(boost::ut::eq(a != b, a_gmp != b_gmp));
        expect(boost::ut::eq(-a, -a_gmp));
        expect(neq(a + 1, a_gmp + 2));
    } | gen_random_compound_int_pairs<4>();
#endif

    return 0;
}
